import sys

from mxnet.initializer import Constant
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock



class Activation(HybridBlock):
    def __init__(self, act_type, channels=1, **kwargs):
        super(Activation, self).__init__(**kwargs)
        with self.name_scope():
            if act_type == 'prelu':
                self.activation = PReLU(channels=channels, prefix='prelu_')
            elif act_type == 'relu':
                self.activation = nn.Activation(act_type, prefix='relu_')
            elif act_type == 'swish':
                self.activation = nn.Swish(beta=1.0, prefix='swish_')
            else:
                sys.exit('Wrong activation type: %s' % act_type)

    def hybrid_forward(self, F, x):
        x = self.activation(x)
        return x



class PReLU(HybridBlock):
    def __init__(self, channels, alpha_initializer=Constant(0.25), **kwargs):
        super(PReLU, self).__init__(**kwargs)
        with self.name_scope():
            self.alpha = self.params.get('alpha',
                                         shape=(channels,),
                                         init=alpha_initializer)

    def hybrid_forward(self, F, x, alpha):
        return F.LeakyReLU(x, gamma=alpha, act_type='prelu', name='fwd')
