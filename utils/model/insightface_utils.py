import sys

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

from utils.model.activation import Activation


class ConvBN(HybridBlock):
    def __init__(self, num_filter, in_channels,
                 kernel=3, stride=1, padding=1, groups=1, use_bias=True,
                 bn_eps=2e-5, bn_mom=0.9, bn_test=False, **kwargs):
        super(ConvBN, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(in_channels=in_channels, channels=num_filter,
                                  kernel_size=kernel, strides=stride,
                                  padding=padding, groups=groups,
                                  use_bias=use_bias, prefix='')
            self.bn = nn.BatchNorm(epsilon=bn_eps, momentum=bn_mom,
                                   use_global_stats=bn_test, prefix='bn_')

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBlock(HybridBlock):
    def __init__(self, num_filter, in_channels,
                 kernel=3, stride=1, padding=1, groups=1,
                 use_bias=True, act_type='prelu',
                 bn_eps=2e-5, bn_mom=0.9, bn_test=False,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv_bn = ConvBN(num_filter, in_channels=in_channels,
                                  kernel=kernel, stride=stride,
                                  padding=padding, groups=groups,
                                  use_bias=use_bias, 
                                  bn_eps=bn_eps, bn_mom=bn_mom,
                                  bn_test=bn_test, prefix='')
            self.activation = Activation(act_type,
                                         channels=num_filter,
                                         prefix='')

    def hybrid_forward(self, F, x):
        x = self.conv_bn(x)
        x = self.activation(x)
        return x


class OutputUnit(HybridBlock):
    def __init__(self, output_type, in_channels, embedding_dim=512,
                 bn_eps=2e-5, bn_mom=0.9, bn_test=False, **kwargs):
        super(OutputUnit, self).__init__(**kwargs)
        bn_params = {
            'epsilon': bn_eps,
            'momentum': bn_mom,
            'use_global_stats': bn_test
        }
        fc_params = {
            'in_units': in_channels * 7 * 7,
            'units': embedding_dim,
        }
        with self.name_scope():
            self.output_unit = nn.HybridSequential(prefix='')
            if output_type == 'E':
                self.output_unit.add(nn.BatchNorm(in_channels=in_channels, 
                                                  prefix='bn1_', **bn_params))
                self.output_unit.add(nn.Dropout(rate=0.4, prefix='dropout_'))
                self.output_unit.add(nn.Dense(prefix='pre_fc1_', **fc_params))
                self.output_unit.add(nn.BatchNorm(in_channels=embedding_dim,
                                                  scale=False, prefix='fc1_',
                                                  **bn_params))
            elif output_type == 'F':
                self.output_unit.add(nn.BatchNorm(in_channels=in_channels,
                                                  prefix='bn1_', **bn_params))
                self.output_unit.add(nn.Dropout(rate=0.4, prefix='dropout_'))
                self.output_unit.add(nn.Dense(prefix='fc1_', **fc_params))
            elif output_type == 'G':
                self.output_unit.add(nn.BatchNorm(in_channels=in_channels,
                                                  prefix='bn1_', **bn_params))
                self.output_unit.add(nn.Dense(prefix='fc1_', **fc_params))
            elif output_type == 'H':
                self.output_unit.add(nn.Dense(prefix='fc1_', **fc_params))
            elif output_type == 'I':
                self.output_unit.add(nn.BatchNorm(in_channels=in_channels,
                                                  prefix='bn1_', **bn_params))
                self.output_unit.add(nn.Dense(prefix='pre_fc1_', **fc_params))
                self.output_unit.add(nn.BatchNorm(in_channels=embedding_dim,
                                                  scale=False, prefix='fc1_',
                                                  **bn_params))
            elif output_type == 'J':
                self.output_unit.add(nn.Dense(prefix='pre_fc1_', **fc_params))
                self.output_unit.add(nn.BatchNorm(in_channels=embedding_dim,
                                                  scale=False, prefix='fc1_',
                                                  **bn_params))
            elif output_type == 'JS':
                self.output_unit.add(nn.Dense(prefix='pre_fc1_', **fc_params))
                self.output_unit.add(nn.BatchNorm(in_channels=embedding_dim,
                                                  scale=True, prefix='fc1_',
                                                  **bn_params))
            else:
                sys.exit('Unsupported output type: %s' % output_type)

    def hybrid_forward(self, F, x):
        x = self.output_unit(x)
        return x
