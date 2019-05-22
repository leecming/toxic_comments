"""
Modified version of code from https://github.com/kpot/keras-transformer
"""

# noinspection PyPep8Naming
# pylint: disable=no-name-in-module
from tensorflow.python.keras import backend as K


class MaskedBinaryCrossedentropy:
    """
    Binary cross-entropy with masking ->
    y_true needs to be [:, 6, 2] where [:, :, 0] contains the labels
    and [:, :, 1] contains the mask values 0/1
    """
    def __init__(self):
        self.__name__ = 'MaskedBinaryCrossedentropy'

    def __call__(self, y_true, y_pred):
        # Keras checks target/pred shapes so need to fake reshape
        y_true = K.reshape(y_true, [-1, 6, 2])
        y_true_val = y_true[:, :, 0]
        mask = y_true[:, :, 1]

        # masked per-sample means of each loss
        num_items_masked = K.sum(mask, axis=-1) + 1e-6
        masked_cross_entropy = (
            K.sum(mask * K.binary_crossentropy(y_true_val, y_pred),
                  axis=-1)
            / num_items_masked)
        return masked_cross_entropy


class MaskedPenalizedSparseCategoricalCrossentropy:
    """
    Masked cross-entropy (see `masked_perplexity` for more details)
    loss function with penalized confidence.
    Combines two loss functions: cross-entropy and negative entropy
    (weighted by `penalty_weight` parameter), following paper
    "Regularizing Neural Networks by Penalizing Confident Output Distributions"
    (https://arxiv.org/abs/1701.06548)

    how to use:
    >>> model.compile(
    >>>     optimizer,
    >>>     loss=MaskedPenalizedSparseCategoricalCrossentropy(0.1))
    """
    def __init__(self, penalty_weight: float):
        self.penalty_weight = penalty_weight
        self.__name__ = 'MaskedPenalizedSparseCategoricalCrossentropy'

    def __call__(self, y_true, y_pred):
        y_true_val = y_true[:, :, 0]
        mask = y_true[:, :, 1]

        # masked per-sample means of each loss
        num_items_masked = K.sum(mask, axis=-1) + 1e-6
        masked_cross_entropy = (
            K.sum(mask * K.sparse_categorical_crossentropy(y_true_val, y_pred),
                  axis=-1)
            / num_items_masked)
        masked_entropy = (
            K.sum(mask * -K.sum(y_pred * K.log(y_pred), axis=-1), axis=-1)
            / num_items_masked)
        return masked_cross_entropy - self.penalty_weight * masked_entropy

    def get_config(self):
        """to help when loading/save models"""
        return {
            'penalty_weight': self.penalty_weight
        }
