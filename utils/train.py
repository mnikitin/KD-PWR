import sys, time, logging
import mxnet as mx
from mxnet import gluon, autograd

from utils.data_utils import get_rec_data_iterators, unpack_batch
from utils.evaluate import load_bin, test

from utils.model import get_model
from utils.loss import get_angular_classifier, get_losses, init_eval_metrics


def train_net(cfg):
    data_source = cfg['data_source']
    data_params = cfg['data_params']
    teacher_params = cfg['teacher_params']
    student_params = cfg['student_params']
    loss_params = cfg['loss_params']
    test_params = cfg['test_params']
    opt_params = cfg['opt_params']

    # Set data iterators
    train_iter, _ = get_rec_data_iterators(data_source['params']['train_db'], '',
                                           cfg['input_shape'], cfg['batch_size'],
                                           data_params, cfg['devices_id'])

    devices = [mx.gpu(device_id) for device_id in cfg['devices_id']]
    batch_size = cfg['batch_size'] * len(devices)
    num_batches = data_source['train_samples_num'] // batch_size

    # Set teacher extractor
    if teacher_params['type'] == 'insightface-resnet':
        teacher_net = get_model(teacher_params, True)
    else:
        sys.exit('Unsupported teacher net architecture: %s !' % teacher_params['type'])
    # Init teacher extractor
    teacher_net.load_parameters('%s-%04d.params' % teacher_params['init'], ctx=devices,
                                allow_missing=False, ignore_extra=False)
    logging.info("Teacher extractor parameters were successfully loaded")
    teacher_net.hybridize(static_alloc=True, static_shape=True)

    # Set student extractor
    if student_params['type'] == 'insightface-resnet':
        student_net = get_model(student_params, False, 'student_')
    else:
        sys.exit('Unsupported student net architecture: %s !' % student_params['type'])
    # Init student extractor
    if student_params['init']:
        student_net.load_parameters('%s-%04d.params' % student_params['init'], ctx=devices,
                                    allow_missing=False, ignore_extra=False)
        logging.info("Student extractor parameters were successfully loaded")
    else:
        init_params = [
            ('.*gamma|.*alpha|.*running_mean|.*running_var', mx.init.Constant(1)),
            ('.*beta|.*bias', mx.init.Constant(0.0)),
            ('.*weight', mx.init.Xavier())
        ]
        for mask, initializer in init_params:
            student_net.collect_params(mask).initialize(initializer, ctx=devices)
    student_net.hybridize(static_alloc=True, static_shape=True)
    
    params = student_net.collect_params()

    # Set teacher classifier
    teacher_clf = None
    if loss_params['HKD']['weight'] > 0.0:
        teacher_clf = get_angular_classifier(data_source['num_classes'],
                                             teacher_params['embedding_dim'],
                                             loss_params['classification'])
        # init teacher classifier
        filename = '%s-%04d.params' % loss_params['HKD']['teacher_init']
        teacher_clf.load_parameters(filename, ctx=devices,
                                    allow_missing=False,
                                    ignore_extra=False)
        logging.info("Teacher classifier parameters "
                     "were successfully loaded")
        teacher_clf.hybridize(static_alloc=True, static_shape=True)
        
    # Set student classifier
    student_clf = None
    if loss_params['HKD']['weight'] > 0.0 or loss_params['classification']['weight'] > 0.0:
        student_clf = get_angular_classifier(data_source['num_classes'],
                                             student_params['embedding_dim'],
                                             loss_params['classification'],
                                             'student_')
        # init student classifier
        if loss_params['classification']['student_init']:
            filename = '%s-%04d.params' % loss_params['classification']['student_init']
            student_clf.load_parameters(filename, ctx=devices,
                                        allow_missing=False,
                                        ignore_extra=False)
            logging.info("Student classifier parameters "
                         "were successfully loaded")
        else:
            student_clf.initialize(mx.init.Normal(0.01), ctx=devices)
        student_clf.hybridize(static_alloc=True, static_shape=True)
        params.update(student_clf.collect_params())

    # Set losses
    L_clf, L_hkd, L_mld = get_losses(loss_params)

     # Set train evaluation metrics
    eval_metrics_train = init_eval_metrics(loss_params)

    # Set optimizer
    optimizer = 'sgd'
    optimizer_params = {'wd': opt_params['wd'], 'momentum': opt_params['momentum']}

    # Set trainer
    trainer = gluon.Trainer(params, optimizer, optimizer_params, kvstore='local')

    # Initialize test results
    test_best_result = {db_name : [0.0, 0] for db_name in test_params['dbs']}

    # TRAINING LOOP
    iteration = 0
    for epoch in range(opt_params['num_epoch']):
        tic_epoch = time.time()

        # reset metrics
        for metric in eval_metrics_train.values():
            metric['metric'].reset()
            metric['losses'] = []

        # update learning rate: step decay
        if epoch == 0:
            trainer.set_learning_rate(opt_params['lr_base'])
        elif epoch > 0 and not epoch % opt_params['lr_epoch_step']:
            trainer.set_learning_rate(trainer.learning_rate * opt_params['lr_factor'])
            logging.info("Learning rate has been changed to %f" % trainer.learning_rate)

        tic_batch = time.time()
        for i, batch in enumerate(train_iter):
            iteration += 1
            # process batch
            data, label = unpack_batch(batch)
            loss = []
            for X, y_gt in zip(data, label):
                # get teacher predictions
                with autograd.predict_mode():
                    embeddings_teacher = teacher_net(X)
                    if teacher_clf:
                        logits_teacher = teacher_clf(embeddings_teacher, y_gt)
                # get student predictions and compute loss
                with autograd.record():
                    embeddings_student = student_net(X)
                    if student_clf:
                        logits_student = student_clf(embeddings_student, y_gt)
                    device_losses = []
                    # classification loss
                    if L_clf:
                        loss_clf = loss_params['classification']['weight'] * \
                                   L_clf(logits_student, y_gt)
                        device_losses.append(loss_clf)
                        eval_metrics_train['classification']['losses'].append(loss_clf)
                    # Hinton's knowledge distillation loss
                    if L_hkd:
                        loss_hkd = loss_params['HKD']['weight'] * \
                                   L_hkd(logits_student, logits_teacher)
                        device_losses.append(loss_hkd)
                        eval_metrics_train['HKD']['losses'].append(loss_hkd)
                    # metric learning distillation losses
                    for name, L, weight in L_mld:
                        loss_mld = weight * L(embeddings_student, embeddings_teacher)
                        device_losses.append(loss_mld)
                        eval_metrics_train[name]['losses'].append(loss_mld)
                    # aggregate all losses
                    device_losses = [loss_term.mean() for loss_term in device_losses]
                    loss.append(mx.nd.add_n(*device_losses))
            eval_metrics_train['total']['losses'] = loss

            # Backpropagate errors
            for l in loss:
                l.backward()
            trainer.step(batch_size)

            # update metrics
            for metric in eval_metrics_train.values():
                metric['metric'].update(_, metric['losses'])
                metric['losses'] = []

            # display training statistics
            if not (i+1) % cfg['display_period']:
                disp_template = 'Epoch[%d/%d] Batch[%d/%d]\tSpeed: %f samples/sec\tlr=%f'
                disp_params = [epoch, opt_params['num_epoch'], i+1, num_batches,
                               batch_size * cfg['display_period'] / (time.time() - tic_batch),
                               trainer.learning_rate]
                for metric in eval_metrics_train.values():
                    metric_name, metric_score = metric['metric'].get()
                    disp_template += '\t%s=%f'
                    disp_params.append(metric_name)
                    disp_params.append(metric_score)
                logging.info(disp_template % tuple(disp_params))
                tic_batch = time.time()

            if not iteration % cfg['test_period']:
                period_idx = iteration // cfg['test_period']
                # save model
                logging.info("[Epoch %d][Batch %d] "
                             "Saving network params [%d] at %s" % 
                            (epoch, i, period_idx, cfg['experiment_dir']))
                student_net.export('%s/student' % cfg['experiment_dir'], period_idx)
                if student_net:
                    student_clf.export('%s/student-clf' % cfg['experiment_dir'], period_idx)
                # test model using outside data
                if test_params['dbs']:
                    logging.info('[Epoch %d] Testing student network ...' % epoch)
                    # emore bin-files testing
                    for db_name in test_params['dbs']:
                        db_path = '%s/%s.bin' % (test_params['dbs_root'], db_name)
                        data_set = load_bin(db_path, [cfg['input_shape'][1], cfg['input_shape'][2]])
                        _, _, acc, std, _, _ = test(data_set, student_net, cfg['batch_size'], 10)
                        if acc > test_best_result[db_name][0]:
                            test_best_result[db_name] = [acc, period_idx]
                        logging.info("Epoch[%d] Batch[%d] %s: "
                                     "Accuracy-Flip = %1.5f+-%1.5f "
                                     "(best: %f, at snapshot %04d)" %
                                    (epoch, i+1, db_name, acc, std,
                                     test_best_result[db_name][0],
                                     test_best_result[db_name][1]))

        # estimate epoch training speed
        throughput = int(batch_size * (i+1) / (time.time() - tic_epoch))
        logging.info("[Epoch %d] Speed: %d samples/sec\t"
                     "Time cost: %f seconds" %
                     (epoch, throughput, time.time() - tic_epoch))
