from mxnet.gluon.loss import Loss


class RelativeDistanceLoss(Loss):
    ''' Relational Knowledge Distillation: distance-wise distillation '''
    def __init__(self, metric_func, mean_normalize=True, huber_delta=1.0,
                 weight=None, batch_axis=0, **kwargs):
        super(RelativeDistanceLoss, self).__init__(weight, batch_axis, **kwargs)
        self.metric_func = metric_func.lower()
        assert self.metric_func in ['l2', 'cs']
        self.huber_delta = huber_delta
        self.mean_normalize = mean_normalize

    def _match(self, F, desc, eps=1e-12):
        prod = F.linalg.syrk(desc, transpose=False)
        if self.metric_func == 'l2':
            # euclidean distance
            squared_desc = F.square(desc)
            sum_squared = F.sum(squared_desc,
                                axis=self._batch_axis, exclude=True)
            l2_squared = F.expand_dims(sum_squared, axis=1) + \
                         F.expand_dims(sum_squared, axis=0) - 2 * prod
            scores = F.sqrt(F.maximum(l2_squared, eps))
        else:
            # inversed cosine distance
            scores = 1.0 - prod
        # construct new distance matrix with zero diag
        scores_upper = F.linalg.extracttrian(scores, offset=1)
        scores_lower = F.linalg.extracttrian(scores, offset=-1)
        scores = F.linalg.maketrian(scores_upper, offset=1) + \
                 F.linalg.maketrian(scores_lower, offset=-1)
        # batch-mean normalization
        if self.mean_normalize:
            mean = F.mean(scores_upper)
            scores = scores / (mean + 1e-8)
        return scores

    def _compute_loss(self, F, pred, label):
        # Huber loss
        abs_dist = F.abs(pred - label)
        return F.where(F.lesser_equal(abs_dist, self.huber_delta),
                       0.5 * F.square(abs_dist),
                       self.huber_delta * (abs_dist - 0.5 * self.huber_delta)
                      )

    def hybrid_forward(self, F, student, teacher):
        student = F.L2Normalization(student, mode='instance')
        teacher = F.L2Normalization(teacher, mode='instance')
        # compute metrics
        student_scores = self._match(F, student)
        teacher_scores = self._match(F, teacher)
        # compute loss
        loss = self._compute_loss(F, student_scores, teacher_scores)
        # average loss
        loss = loss.sum(axis=self._batch_axis, exclude=True) / \
               max(loss.shape[self._batch_axis] - 1, 1)
        return loss


class RelativeAngleLoss(Loss):
    ''' Relational Knowledge Distillation: angle-wise distillation '''
    def __init__(self, huber_delta=1.0,
                 weight=None, batch_axis=0, **kwargs):
        super(RelativeAngleLoss, self).__init__(weight, batch_axis, **kwargs)
        self.huber_delta = huber_delta

    def _compute_angle_distance(self, F, desc):
        diff = F.expand_dims(desc, axis=0) - F.expand_dims(desc, axis=1)
        diff_norm = F.L2Normalization(diff, mode='spatial')
        scores = F.linalg.syrk(diff_norm, transpose=False)
        return scores

    def _compute_loss(self, F, pred, label):
        # Huber loss
        abs_dist = F.abs(pred - label)
        return F.where(F.lesser_equal(abs_dist, self.huber_delta),
                       0.5 * F.square(abs_dist),
                       self.huber_delta * (abs_dist - 0.5 * self.huber_delta)
                      )

    def hybrid_forward(self, F, student, teacher):
        student = F.L2Normalization(student, mode='instance')
        teacher = F.L2Normalization(teacher, mode='instance')
        # compute metrics
        student_scores = self._compute_angle_distance(F, student)
        teacher_scores = self._compute_angle_distance(F, teacher)
        # compute loss
        loss = self._compute_loss(F, student_scores, teacher_scores)
        loss = loss.mean(axis=self._batch_axis, exclude=True)
        return loss
