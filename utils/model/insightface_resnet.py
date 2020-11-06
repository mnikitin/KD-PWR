import sys

import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

from utils.model.insightface_utils import ConvBN, ConvBlock, OutputUnit


class ResidualBlock(HybridBlock):
    def __init__(self, num_filter, in_channels,
                 stride, dim_match, bottleneck, act_type='prelu',
                 bn_eps=2e-5, bn_mom=0.9, bn_test=False,
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.num_filter = num_filter
        self.in_channels = in_channels
        self.stride = stride
        self.act_type = act_type
        self.dim_match = dim_match
        self.conv_bn_params = {
            'use_bias': False,
            'bn_eps': bn_eps,
            'bn_mom': bn_mom,
            'bn_test': bn_test
        }
        with self.name_scope():
            self.convs = self._make_residual_unit_v3(bottleneck)
            if not dim_match:
                self.shortcut = ConvBN(num_filter,
                                       in_channels=self.in_channels,
                                       kernel=1, stride=stride, padding=0,
                                       prefix='conv_sc_', **self.conv_bn_params)

    def _make_residual_unit_v3(self, bottleneck):
        convs = nn.HybridSequential(prefix='')
        with convs.name_scope():
            convs.add(nn.BatchNorm(epsilon=self.conv_bn_params['bn_eps'],
                                   momentum=self.conv_bn_params['bn_mom'],
                                   use_global_stats=self.conv_bn_params['bn_test'],
                                   prefix='bn1_'))
            if bottleneck:
                convs.add(ConvBlock(int(self.num_filter * 0.25),
                                    in_channels=self.in_channels,
                                    kernel=1, stride=1, padding=0,
                                    act_type=self.act_type,
                                    prefix='conv1_', **self.conv_bn_params))
                convs.add(ConvBlock(int(self.num_filter * 0.25),
                                    in_channels=self.in_channels,
                                    kernel=3, stride=1, padding=1,
                                    act_type=self.act_type,
                                    prefix='conv2_', **self.conv_bn_params))
                convs.add(ConvBN(self.num_filter, kernel=1,
                                 in_channels=self.in_channels,
                                 stride=self.stride, padding=0,
                                 prefix='conv3_', **self.conv_bn_params))
            else:
                convs.add(ConvBlock(self.num_filter,
                                    in_channels=self.in_channels,
                                    kernel=3, stride=1, padding=1,
                                    act_type=self.act_type,
                                    prefix='conv1_', **self.conv_bn_params))
                convs.add(ConvBN(self.num_filter, in_channels=self.num_filter,
                                 kernel=3, stride=self.stride, padding=1,
                                 prefix='conv2_', **self.conv_bn_params))
        return convs

    def hybrid_forward(self, F, x):
        if self.dim_match:
            shortcut = x
        else:
            shortcut = self.shortcut(x)
        x = self.convs(x)
        x = x + shortcut
        return x


class ResNet(HybridBlock):
    def __init__(self, input_type, output_type, act_type,
                 num_res_block, num_conv_map,
                 bottleneck, embedding_dim,
                 bn_eps=2e-5, bn_mom=0.9, bn_test=False,
                 **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.input_type = input_type
        self.output_type = output_type
        self.act_type = act_type
        self.bottleneck = bottleneck
        self.bn_params = {
            'bn_eps': bn_eps,
            'bn_mom': bn_mom,
            'bn_test': bn_test
        }
        with self.name_scope():
            # input
            if input_type == 1:
                self.input_mean = 127.5
                self.input_scale = 1.0 / 128.0
            elif input_type == 2:
                self.input_norm = nn.BatchNorm(in_channels=3,
                                    epsilon=self.bn_params['bn_eps'],
                                    momentum=self.bn_params['bn_mom'],
                                    use_global_stats=self.bn_params['bn_test'],
                                    prefix='bn_')
            else:
                sys.exit('Unsupported input type: %d' % input_type)
            self.input_conv = ConvBlock(num_filter=num_conv_map[0], 
                                        in_channels=3,
                                        kernel=3, stride=1, padding=1,
                                        use_bias=False, act_type=self.act_type,
                                        prefix='conv0_', **self.bn_params)

            # body
            self.stage1_unit = self._make_stage_unit(1, num_res_block[0],
                                        num_conv_map[1], num_conv_map[0])
            self.stage2_unit = self._make_stage_unit(2, num_res_block[1],
                                        num_conv_map[2], num_conv_map[1])
            self.stage3_unit = self._make_stage_unit(3, num_res_block[2],
                                        num_conv_map[3], num_conv_map[2])
            self.stage4_unit = self._make_stage_unit(4, num_res_block[3],
                                        num_conv_map[4], num_conv_map[3])
            if bottleneck:
                self.bottleneck_unit = ConvBlock(num_filter=512,
                                                 in_channels=num_conv_map[4],
                                                 kernel=1, stride=1, padding=0,
                                                 use_bias=False,
                                                 act_type=self.act_type, 
                                                 prefix='conv_btlnck_',
                                                 **self.bn_params)
            # output
            self.output_unit = OutputUnit(output_type, num_conv_map[4],
                                          embedding_dim=embedding_dim,
                                          prefix='output_', **self.bn_params)

    def _make_stage_unit(self, stage_idx, units_num, filters_num, in_channels):
        stage_unit = nn.HybridSequential(prefix='stage%d_' % stage_idx)
        with stage_unit.name_scope():
            stage_unit.add(ResidualBlock(num_filter=filters_num,
                                         in_channels=in_channels,
                                         stride=2, dim_match=False,
                                         bottleneck=self.bottleneck,
                                         act_type=self.act_type,
                                         **self.bn_params, prefix='unit1_'))
            for idx in range(units_num - 1):
                stage_unit.add(ResidualBlock(num_filter=filters_num,
                                             in_channels=filters_num,
                                             stride=1, dim_match=True,
                                             bottleneck=self.bottleneck,
                                             act_type=self.act_type,
                                             prefix='unit%d_' % (idx+2),
                                             **self.bn_params))
        return stage_unit

    def hybrid_forward(self, F, x):
        # input
        if self.input_type == 1:
            x = (x - self.input_mean) * self.input_scale
        elif self.input_type == 2:
            x = self.input_norm(x)
        x = self.input_conv(x)
        # body
        x = self.stage1_unit(x)
        x = self.stage2_unit(x)
        x = self.stage3_unit(x)
        x = self.stage4_unit(x)
        if self.bottleneck:
            x = self.bottleneck_unit(x)
        # output
        x = self.output_unit(x)
        return x


def get_insightface_resnet(depth, act_type='prelu',
                           input_type=1, output_type='G',
                           embedding_dim=512,
                           bn_use_global_stats=False,
                           prefix=''):
    if depth >= 500:
        num_conv_map = [64, 256, 512, 1024, 2048]
        bottleneck = True
    else:
        num_conv_map = [64, 64, 128, 256, 512]
        bottleneck = False
    if depth == 18:
        num_res_block = [2, 2, 2, 2]
    elif depth == 34:
        num_res_block = [3, 4, 6, 3]
    elif depth == 49:
        num_res_block = [3, 4, 14, 3]
    elif depth == 50:
        num_res_block = [3, 4, 14, 3]
    elif depth == 74:
        num_res_block = [3, 6, 24, 3]
    elif depth == 90:
        num_res_block = [3, 8, 30, 3]
    elif depth == 98:
        num_res_block = [3, 4, 38, 3]
    elif depth == 99:
        num_res_block = [3, 8, 35, 3]
    elif depth == 100:
        num_res_block = [3, 13, 30, 3]
    elif depth == 134:
        num_res_block = [3, 10, 50, 3]
    elif depth == 136:
        num_res_block = [3, 13, 48, 3]
    elif depth == 140:
        num_res_block = [3, 15, 48, 3]
    elif depth == 124:
        num_res_block = [3, 13, 40, 5]
    elif depth == 160:
        num_res_block = [3, 24, 49, 3]
    elif depth == 101:
        num_res_block = [3, 4, 23, 3]
    elif depth == 152:
        num_res_block = [3, 8, 36, 3]
    elif depth == 200:
        num_res_block = [3, 24, 36, 3]
    elif depth == 269:
        num_res_block = [3, 30, 48, 8]
    else:
        sys.exit("No experiments done on depth %d. "
                 "You can do it yourself." % depth)

    model = ResNet(input_type=input_type,
                   output_type=output_type,
                   act_type=act_type,
                   num_res_block=num_res_block,
                   num_conv_map=num_conv_map,
                   bottleneck=bottleneck,
                   embedding_dim=embedding_dim,
                   bn_test=bn_use_global_stats,
                   prefix=prefix
                   )
    return model
