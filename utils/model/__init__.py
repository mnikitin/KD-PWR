import sys

from utils.model.insightface_resnet import get_insightface_resnet


def get_model(params: dict, bn_test: bool, prefix=''):
    if params['type'] == 'insightface-resnet':
        net = get_insightface_resnet(params['depth'],
                                     act_type=params['activation'],
                                     input_type=params['input_type'],
                                     output_type=params['output_type'],
                                     embedding_dim=params['embedding_dim'],
                                     bn_use_global_stats=bn_test,
                                     prefix=prefix)
    else:
        sys.exit('Unsupported net architecture: %s' % params['type'])
    return net
