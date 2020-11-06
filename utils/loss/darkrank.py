from mxnet.gluon.loss import Loss

from itertools import permutations


class DarkRankLoss(Loss):
    ''' DarkRank Loss: hard (ListMLE) or soft (ListNet) '''
    def __init__(self, metric_func='l2', loss_type='hard',
                 alpha=3.0, beta=3.0, list_length=4,
                 weight=None, batch_axis=0, **kwargs):
        super(DarkRankLoss, self).__init__(weight, batch_axis, **kwargs)
        self.metric_func = metric_func.lower()
        assert self.metric_func in ['l2', 'cs']
        self.type = loss_type.lower()
        assert self.type in ['hard', 'soft']
        self.alpha = alpha
        self.beta = beta
        self.list_length = list_length
        assert self.list_length > 1

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
        # augment scores
        scores = -self.alpha * F.power(scores, self.beta)
        return scores

    def _compute_loss_hard(self, F, pred, label):
        batch_size = pred.shape[0]
        list_len = min(self.list_length, batch_size)
        rank = F.slice_axis(F.argsort(-label), axis=1, begin=1, end=list_len)
        offset = F.reshape(F.arange(0, batch_size ** 2, batch_size),
                           (batch_size, 1))
        rank_flat = F.reshape(F.broadcast_add(rank, offset), (-1,))
        # index student distances with teacher ranking
        pred_flat = F.reshape(pred, (-1,))
        pred_ranked = F.reshape_like(F.take(pred_flat, rank_flat), rank)
        # compute loss
        loss = []
        for idx in range(list_len - 1):
            pred_ranked_slice = F.slice_axis(pred_ranked, axis=1,
                                             begin=idx, end=list_len-1)
            log_softmax = F.log_softmax(pred_ranked_slice, axis=1)
            loss.append(F.slice_axis(log_softmax, axis=1, begin=0, end=1))
        loss_sum = F.add_n(*loss)
        loss = -loss_sum.mean()
        return loss

    def _compute_loss_soft(self, F, pred, label):
        batch_size = pred.shape[0]
        list_len = min(self.list_length, batch_size)
        rank = F.slice_axis(F.argsort(-label), axis=1, begin=1, end=batch_size)
        offset = F.reshape(F.arange(0, batch_size ** 2, batch_size),
                           (batch_size, 1))
        rank_flat = F.reshape(F.broadcast_add(rank, offset), (-1,))
        # index student distances with teacher ranking
        pred_flat = F.reshape(pred, (-1,))
        pred_ranked = F.reshape_like(F.take(pred_flat, rank_flat), rank)
        pred_ranked = F.slice_axis(pred_ranked, axis=1,
                                   begin=0, end=list_len)
        # index teacher distances with teacher ranking
        label_flat = F.reshape(label, (-1,))
        label_ranked = F.reshape_like(F.take(label_flat, rank_flat), rank)
        label_ranked = F.slice_axis(label_ranked, axis=1,
                                    begin=0, end=list_len)
        # compute loss
        loss = []
        for perm in permutations(range(list_len)):
            mx_perm = F.BlockGrad(F.array(perm))
            pred_perm = F.take(pred_ranked, mx_perm, axis=1)
            label_perm = F.take(label_ranked, mx_perm, axis=1)
            perm_student_logprob = []
            perm_teacher_logprob = []
            for idx in range(list_len - 1):
                pred_slice = F.slice_axis(pred_perm, axis=1,
                                          begin=idx, end=list_len)
                pred_log_softmax = F.log_softmax(pred_slice, axis=1)
                perm_student_logprob.append(
                    F.slice_axis(pred_log_softmax, axis=1, begin=0, end=1))
                label_slice = F.slice_axis(label_perm, axis=1,
                                           begin=idx, end=list_len)
                label_log_softmax = F.log_softmax(label_slice, axis=1)
                perm_teacher_logprob.append(
                    F.slice_axis(label_log_softmax, axis=1, begin=0, end=1))
            student_logprob = F.add_n(*perm_student_logprob)
            teacher_logprob = F.add_n(*perm_teacher_logprob)
            teacher_prob = F.exp(perm_teacher_logprob)
            kl_div = teacher_prob * (teacher_logprob - student_logprob)
            loss.append(kl_div)
        loss_sum = F.add_n(*loss)
        loss = loss_sum.mean()
        return loss

    def hybrid_forward(self, F, student, teacher):
        student = F.L2Normalization(student, mode='instance')
        teacher = F.L2Normalization(teacher, mode='instance')
        # compute metrics
        student_scores = self._match(F, student)
        teacher_scores = self._match(F, teacher)
        # compute loss
        if self.type == 'hard':
            loss = self._compute_loss_hard(F, student_scores, teacher_scores)
        else:
            loss = self._compute_loss_soft(F, student_scores, teacher_scores)
        return loss