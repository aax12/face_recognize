from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
import tensorflow as tf


def hard_triplet_loss(distance, pos_mask, neg_mask, margin):
    pos_distance = distance * pos_mask

    hardest_pos_distance = K.max(pos_distance, -1, True)

    max_distance = K.max(distance, -1, True)
    hardest_neg_distance = K.min(distance * neg_mask + max_distance * (1. - neg_mask), -1, True)

    return K.mean(K.maximum(hardest_pos_distance - hardest_neg_distance + margin, 0.))


def semi_triplet_loss(distance, pos_mask, neg_mask, margin):
    pos_distance = distance * pos_mask

    hardest_pos_distance = K.max(pos_distance, -1, True)
    semi_neg_mask = K.cast(K.greater(distance * neg_mask, hardest_pos_distance), 'float32')

    max_distance = K.max(distance, -1, True)
    semi_neg_distance = K.min(distance * semi_neg_mask + max_distance * (1. - semi_neg_mask), -1, True)

    return K.maximum(hardest_pos_distance - semi_neg_distance + margin, 0.)


class OnlineTripletLoss(Loss):
    def __init__(self, margin, method):
        self.margin = margin
        self.method = method
        super(OnlineTripletLoss, self).__init__()

    def call(self, y_true, y_pred):
        '''
        y_true: [batch, 1]
        y_pred: [batch, 128]
        '''
        labels = K.cast(K.squeeze(y_true, -1), 'int32')     # [batch]

        # tf.print(K.shape(y_true), K.shape(y_pred))

        distance = self._pairwise_distance(y_pred)

        positive_mask = self._get_positive_mask(labels)
        negative_mask = self._get_negative_mask(labels)

        loss = self.method(distance, positive_mask, negative_mask, self.margin)

        return loss

    def _pairwise_distance(self, embed):
        '''
        embeded: [batches, 128]
        '''
        rows_embed = K.expand_dims(embed, 0)     # [1, batch, 128]
        cols_embed = K.expand_dims(embed, 1)     # [batch, 1, 128]

        pair_norm = K.sum(K.square(rows_embed - cols_embed), -1)     # [batch, batch]

        '''
        ||a - b|| ^ 2 = ||a|| ^ 2 - 2 * (a * b) + ||b|| ^ 2
        '''

        zero_mask = K.cast(K.equal(0., pair_norm), 'float32')
        pair_norm += zero_mask * 1e-16
        pair_distance = K.sqrt(pair_norm)

        return pair_distance * (1.0 - zero_mask)

    def _get_positive_mask(self, label):
        '''
        not_equal_mask = 1 - tf.eye(tf.shape(label)[0])   # [batches, batches]
        positive_mask = K.cast(K.equal(K.expand_dims(label, -1), K.expand_dims(label, 0)), 'float32')

        return positive_mask * not_equal_mask
        '''
        indices_not_equal = 1.0 - tf.eye(K.shape(label)[0])
        labels_equal = K.cast(K.equal(K.expand_dims(label, 0), K.expand_dims(label, -1)), 'float32')

        return indices_not_equal * labels_equal

    def _get_negative_mask(self, label):
        return K.cast(K.not_equal(K.expand_dims(label, 0), K.expand_dims(label, -1)), 'float32')


class batch_all_triplet_loss(Loss):
    def __init__(self, margin):
        self.margin = margin
        super(batch_all_triplet_loss, self).__init__()

    def _pairwise_distance(self, embed):
        '''
        embeded: [batches, 128]
        '''
        rows_embed = K.expand_dims(embed, 0)     # [1, batch, 128]
        cols_embed = K.expand_dims(embed, 1)     # [batch, 1, 128]

        pair_norm = K.sum(K.square(rows_embed - cols_embed), -1)     # [batch, batch]

        '''
        ||a - b|| ^ 2 = ||a|| ^ 2 - 2 * (a * b) + ||b|| ^ 2
        '''

        zero_mask = K.cast(K.equal(0., pair_norm), 'float32')
        pair_norm += zero_mask * 1e-16
        pair_distance = K.sqrt(pair_norm)

        return pair_distance * (1.0 - zero_mask)

    def _get_triplet_mask(self, labels):
        indices_not_equal = 1. - tf.eye(K.shape(labels)[0])

        i_not_equal_j = K.expand_dims(indices_not_equal, 2)
        i_not_equal_k = K.expand_dims(indices_not_equal, 1)
        j_not_equal_k = K.expand_dims(indices_not_equal, 0)

        distint_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

        labels_equal = K.cast(K.equal(K.expand_dims(labels, 0), K.expand_dims(labels, 1)), 'float32')
        i_equal_j = K.expand_dims(labels_equal, 2)
        i_equal_k = K.expand_dims(labels_equal, 1)

        valid_labels = i_equal_j * i_equal_k

        return distint_indices * valid_labels

    def call(self, y_true, y_pred):
        labels = K.cast(y_true, 'int32')
        pairwise_distance = self._pairwise_distance(y_pred)

        anchor_positive_dist = K.expand_dims(pairwise_distance, 2)      # [batch, batch, 1, 128]
        anchor_negative_dist = K.expand_dims(pairwise_distance, 1)      # [batch, 1, batch, 128]

        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        mask = self._get_triplet_mask(labels)
        triplet_loss = K.maximum(triplet_loss * mask, 0.)

        valid_triplet = K.cast(K.greater(triplet_loss, 1e-16), 'float32')

        return K.sum(triplet_loss) / (K.sum(valid_triplet) + 1e-16)


class metric(Metric):
    def __init__(self, threshold):
        super(metric, self).__init__()

        self.threshold = threshold
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.squeeze(y_true, -1)
        y_pred = K.squeeze(y_pred, -1)

        valid_mask = K.cast(K.equal(K.cast(y_true, 'bool'), y_pred < self.threshold), 'float32')

        self.correct.assign_add(K.sum(valid_mask))
        self.count.assign_add(K.cast(K.shape(valid_mask)[0], 'float32'))

    def result(self):
        return self.correct / self.count


class BalancedBCE(Loss):
    def __init__(self):
        super(BalancedBCE, self).__init__()

    def call(self, y_true, y_pred):
        mean_positive = K.mean(y_true, 0)
        loss = y_true * (1. - mean_positive) * K.log(y_pred) + (1. - y_true) * mean_positive * K.log(1. - y_pred)

        return -K.mean(loss)


class triplet_loss(Loss):
    def __init__(self, margin):
        self.margin = margin

        super(triplet_loss, self).__init__()

    def call(self, y_true, y_pred):
        a_vector = y_pred[:, 0]
        p_vector = y_pred[:, 1]
        n_vector = y_pred[:, 2]

        ap_distance = K.sqrt(K.sum(K.square(a_vector - p_vector), -1))
        an_distance = K.sqrt(K.sum(K.square(a_vector - n_vector), -1))

        loss = K.maximum(ap_distance - an_distance + self.margin, 0.)
        loss = K.mean(loss) * 2.

        return loss

































