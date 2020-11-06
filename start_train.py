import logging, sys, os, math
import datetime, json

from config import config
from utils.train import train_net


def save_config(cfg):
    save_fname = cfg['experiment_dir'] + '/config.json'
    with open(save_fname, 'w') as f:
        json.dump(cfg, f, sort_keys=True, indent=4)


def set_logging(experiment_dir):
    # duplicate logging to file and stdout
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s]\t%(message)s',
                        datefmt='%m-%d-%y %H:%M',
                        filename=experiment_dir + '/log.txt',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(console)


def print_training_info(cfg):
    batch_size = cfg['batch_size']
    num_devices = len(cfg['devices_id'])
    logging.info("Training info:")
    logging.info("classification loss: {}".format(cfg['loss_params']['classification']['weight']))
    logging.info("HKD loss: {}".format(cfg['loss_params']['HKD']['weight']))
    logging.info("RKD-D loss: {}".format(cfg['loss_params']['RKD-D']['weight']))
    logging.info("RKD-A loss: {}".format(cfg['loss_params']['RKD-A']['weight']))
    logging.info("PWR loss: {}".format(cfg['loss_params']['PWR']['weight']))
    logging.info("total batchsize: {} * {} = {}".format(batch_size, num_devices, batch_size*num_devices))
    logging.info("lr base: {}".format(cfg['opt_params']['lr_base']))
    logging.info("lr epoch step: {}".format(cfg['opt_params']['lr_epoch_step']))
    logging.info("num epoch: {}".format(cfg['epoch_params']['num_epoch']))
    logging.info("total iter num: {}".format(int(cfg['epoch_params']['num_epoch'] *
                                             math.ceil(cfg['data_source']['train_samples_num'] /
                                                       (batch_size * num_devices)))))


def main(argc, argv):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    config['experiment_dir'] += '/' + timestamp
    os.makedirs(config['experiment_dir'])
    save_config(config)
    set_logging(config['experiment_dir'])
    print_training_info(config)
    train_net(config)


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
