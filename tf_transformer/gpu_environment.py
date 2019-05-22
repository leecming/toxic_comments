"""
Modified version of nvidia-optimized BERT code
Helper functions to train in target precision, while
accumulating in float32
"""

import tensorflow as tf


def float32_variable_storage_getter(getter, name, *args, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
       float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, *args, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


def get_custom_getter(compute_type):
    """apply fp16 training only if specifically invoked"""
    return float32_variable_storage_getter if compute_type == tf.float16 else None
