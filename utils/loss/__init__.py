import mxnet as mx

from utils.loss.angular import AngularMarginBlock
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss as CE

from utils.loss.hkd import SoftLogitsLoss
from utils.loss.darkrank import DarkRankLoss
from utils.loss.rkd import RelativeDistanceLoss, RelativeAngleLoss
from utils.loss.pwr import PairwiseRankingLoss



def get_angular_classifier(num_classes: int, embedding_dim: int,
                           params: dict, prefix=''):
    return AngularMarginBlock(num_classes=num_classes,
                              input_dim=embedding_dim,
                              margin_1=params['margin_1'],
                              margin_2=params['margin_2'],
                              margin_3=params['margin_3'],
                              scale=params['scale'],
                              prefix=prefix)


def get_losses(params: dict):
    # Set classification loss
    L_clf = None
    if params['classification']['weight'] > 0.0:
        L_clf = CE(from_logits=False, sparse_label=True)
    # Set Hinton's knowledge distillation loss
    L_hkd = None
    if params['HKD']['weight'] > 0.0:
        L_hkd = SoftLogitsLoss(temperature=params['HKD']['temperature'])
    # Set metric learning distillation losses
    L_mld = []
    if params['DarkRank']['weight'] > 0.0:
        L_dark = DarkRankLoss(metric_func=params['DarkRank']['metric_func'],
                              loss_type=params['DarkRank']['type'],
                              alpha=params['DarkRank']['alpha'],
                              beta=params['DarkRank']['beta'],
                              list_length=params['DarkRank']['list_length'])
        L_mld.append(('DarkRank', L_dark, params['DarkRank']['weight']))
    if params['RKD-D']['weight'] > 0.0:
        L_rkd_d = RelativeDistanceLoss(params['RKD-D']['metric_func'],
                                       mean_normalize=params['RKD-D']['batch_normalize'],
                                       huber_delta=params['RKD-D']['huber_delta'])
        L_mld.append(('RKD-D', L_rkd_d, params['RKD-D']['weight']))
    if params['RKD-A']['weight'] > 0.0:
        L_rkd_a = RelativeAngleLoss(huber_delta=params['RKD-A']['huber_delta'])
        L_mld.append(('RKD-A', L_rkd_a, params['RKD-A']['weight']))
    if params['PWR']['weight'] > 0.0:
        L_pwr = PairwiseRankingLoss(params['PWR']['metric_func'],
                                    params['PWR']['loss_type'],
                                    diff_margin_type=params['PWR']['diff_margin_type'],
                                    diff_margin=params['PWR']['diff_margin'],
                                    diff_power=params['PWR']['diff_power'],
                                    diff_exp_beta=params['PWR']['diff_exp_beta'],
                                    ranknet_beta=params['PWR']['ranknet_beta'])
        L_mld.append(('PWR', L_pwr, params['PWR']['weight']))
    return L_clf, L_hkd, L_mld



def init_eval_metrics(params):
    # Prepare names of metric and loss
    names = []
    if params['classification']['weight'] > 0.0:
        names.append(('classification', 'CE-loss'))
    if params['HKD']['weight'] > 0.0:
        names.append(('HKD', 'HKD-loss'))
    if params['DarkRank']['weight'] > 0.0:
        loss_name = 'DarkRank-%s-%s-loss' % (params['DarkRank']['metric_func'],
                                             params['DarkRank']['type'])
        names.append(('DarkRank', loss_name))
    if params['RKD-D']['weight'] > 0.0:
        loss_name = 'RKD-D-%s-loss' % (params['RKD-D']['metric_func'] + \
                        ('-N' if params['RKD-D']['batch_normalize'] else ""))
        names.append(('RKD-D', loss_name))
    if params['RKD-A']['weight'] > 0.0:
        names.append(('RKD-A', 'RKD-A-loss'))
    if params['PWR']['weight'] > 0.0:
        loss_name = 'PWR-{}'.format(params['PWR']['metric_func'])
        if params['PWR']['loss_type'] == 'ranknet':
            loss_name += '-ranknet-{}'.format(params['PWR']['ranknet_beta'])
        else:
            loss_name += '-diff'
            # add margin info
            if params['PWR']['diff_margin_type'] == 'const':
                loss_name += '-m{}'.format(params['PWR']['loss_params']['diff']['margin'])
            elif params['PWR']['diff_margin_type'] == 'teacher_diff':
                loss_name += '-mTDIFF'
            elif params['PWR']['diff_margin_type'] == 'teacher_std':
                loss_name += '-mTSTD'
            # add loss info
            if params['PWR']['loss_type'] == 'diff_power':
                loss_name += '-power-{}'.format(params['PWR']['diff_power'])
            elif params['PWR']['loss_type'] == 'diff_exp':
                loss_name += '-exp-{}'.format(params['PWR']['diff_exp_beta'])
        loss_name += '-loss'
        names.append(('PWR', loss_name))
    names.append(('total', 'Loss'))
    # Init metrics dict
    eval_metrics = {}
    for metric_name, loss_name in names:
        eval_metrics[metric_name] = {
            'metric': mx.metric.Loss(name=loss_name),
            'losses': []
        }
    return eval_metrics