from mxnet.gluon.loss import Loss


class PairwiseRankingLoss(Loss):
    ''' Pairwise Ranking Distillation '''
    def __init__(self, dist_func, loss_type,
                 diff_margin_type='const', diff_margin=0.0,
                 diff_power=1.0, diff_exp_beta=1.0, ranknet_beta=1.0,
                 weight=None, batch_axis=0, **kwargs):
        super(PairwiseRankingLoss, self).__init__(weight, batch_axis, **kwargs)
        self.dist_func = dist_func.lower()
        assert self.dist_func in ['l2', 'cs']
        self.loss_type = loss_type.lower()
        assert self.loss_type in ['diff_power', 'diff_exp', 'ranknet']
        self.diff_margin_type = diff_margin_type.lower()
        assert self.diff_margin_type in ['const', 'teacher_diff', 'teacher_std']
        self.diff_margin = diff_margin
        self.diff_power = diff_power
        self.diff_exp_beta = diff_exp_beta
        self.ranknet_beta = ranknet_beta

    def _match(self, F, desc, eps=1e-12):
        prod = F.linalg.syrk(desc, transpose=False)
        if self.dist_func == 'l2':
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
        return scores

    def _compute_diff_loss(self, F, diff_pred, diff_label,
                                 label_unique, eps=1e-12):
        # 1. add margin to student scores difference
        if self.diff_margin_type == 'teacher_diff':
            scores_diff = -diff_pred + diff_label
        elif self.diff_margin_type == 'teacher_std':
            _, teacher_var = F.moments(label_unique, axes=0)
            teacher_std = F.sqrt(teacher_var)
            scores_diff = -diff_pred + teacher_std
        else: # const margin
            scores_diff = -diff_pred + self.diff_margin
        # 2. compute difference loss
        if self.loss_type == 'diff_power':
            scores_diff_pos = F.relu(scores_diff) + eps
            diff_loss = F.power(scores_diff_pos, self.diff_power)
        else: # diff_exp
            scores_diff_exp = F.exp(self.diff_exp_beta * scores_diff)
            diff_loss = F.relu(scores_diff_exp - 1.0)
        return diff_loss

    def _compute_loss(self, F, pred, label):
        # get unique scores
        pred_unique = F.linalg.extracttrian(pred, offset=1)
        label_unique = F.linalg.extracttrian(label, offset=1)
        # compute differences between scores
        diff_pred = F.broadcast_minus(F.expand_dims(pred_unique, axis=1),
                                      F.expand_dims(pred_unique, axis=0))
        diff_label = F.broadcast_minus(F.expand_dims(label_unique, axis=1),
                                       F.expand_dims(label_unique, axis=0))
        # compare unique label scores
        is_greater_labels = F.broadcast_greater(diff_label,
                                                F.zeros_like(diff_label))
        # compute loss
        if self.loss_type == 'ranknet':
            diff_loss = F.log(1 + F.exp(-self.ranknet_beta * diff_pred))
        else:
            diff_loss = self._compute_diff_loss(F, diff_pred, diff_label,
                                                   label_unique)
        # get target loss and average it
        loss = F.sum(diff_loss * is_greater_labels) / F.sum(is_greater_labels)
        return loss

    def hybrid_forward(self, F, student, teacher):
        student = F.L2Normalization(student, mode='instance')
        teacher = F.L2Normalization(teacher, mode='instance')
        # compute metrics
        student_scores = self._match(F, student)
        teacher_scores = self._match(F, teacher)
        # compute loss
        loss = self._compute_loss(F, student_scores, teacher_scores)
        return loss
