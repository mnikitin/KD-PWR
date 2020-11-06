from mxnet.gluon.loss import Loss


class SoftLogitsLoss(Loss):
    ''' Hinton's Knowledge Distillation'''
    def __init__(self, axis=-1, temperature=1.0,
                 weight=None, batch_axis=0, **kwargs):
        super(SoftLogitsLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._temperature = temperature

    def hybrid_forward(self, F, student, teacher):
        # raw unnormalized input is assumed
        prob_teacher = F.softmax(teacher, temperature=self._temperature)
        log_prob_student = F.log_softmax(student, self._axis,
                                         temperature=self._temperature)
        log_prob_teacher = F.log_softmax(teacher, self._axis,
                                         temperature=self._temperature)
        kl_div = prob_teacher * (log_prob_teacher - log_prob_student)
        loss = kl_div.sum(axis=self._axis, keepdims=True)
        loss = loss.mean(axis=self._batch_axis, exclude=True)
        return loss




