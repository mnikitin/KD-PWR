from mxnet.gluon.block import HybridBlock
from math import cos, sin


class AngularMarginBlock(HybridBlock):
    """ Combined angular margin block:
        cos(theta)  --->>>  scale * [cos(margin_1 * theta + margin_2) - margin_3]
    """
    def __init__(self, num_classes=1000, input_dim=512,
                 margin_1=1.0, margin_2=0.0, margin_3=0.35, scale=30.0,
                 weight_name='weight', **kwargs):
        super(AngularMarginBlock, self).__init__(**kwargs)
        self.scale = scale
        self.margin_1 = margin_1
        self.margin_2 = margin_2
        self.margin_3 = margin_3
        self.num_classes = num_classes
        with self.name_scope():
            self.fc_weight = self.params.get(weight_name,
                                             shape=(num_classes, input_dim))

    def hybrid_forward(self, F, x, label, fc_weight):
        x_norm = F.L2Normalization(x, mode='instance')
        fc_weight_norm = F.L2Normalization(fc_weight, mode='instance')
        cos_theta = F.FullyConnected(x_norm, fc_weight_norm,
                                     no_bias=True, num_hidden=self.num_classes)
        cos_theta = F.clip(cos_theta, -1.0, 1.0)
        if self.margin_1 != 1.0 or self.margin_2 > 0.0 or self.margin_3 > 0.0:
            gt_one_hot = F.one_hot(label, depth=self.num_classes,
                                   on_value=1.0, off_value=0.0)
            cos_updated = F.pick(cos_theta, label, axis=1, keepdims=True)
            if self.margin_1 != 1.0:    # SphereFace
                theta = F.arccos(cos_updated)
                cos_updated = F.cos(self.margin_1 * theta)
            if self.margin_2 > 0.0:     # ArcFace
                sin_theta = F.sqrt(1.0 - F.square(cos_updated))
                cos_updated = cos_updated * cos(self.margin_2) - \
                              sin_theta * sin(self.margin_2)
            if self.margin_3 > 0.0:     # CosFace
                cos_updated = cos_updated - self.margin_3
            cos_theta = F.broadcast_mul(gt_one_hot, cos_updated) + \
                        F.broadcast_mul(1.0 - gt_one_hot, cos_theta)
        output = self.scale * cos_theta
        return output

