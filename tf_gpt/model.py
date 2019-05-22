"""
Modified version of code from https://github.com/openai/gpt-2
Provides the model code for the transformer decoder architecture
with relevant helper ops
Note that def classifier_model is used for toxic comments
"""

import collections
import re
import numpy as np
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.contrib.training import HParams


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        var_name_matcher = re.match("^(.*):\\d+$", name)
        if var_name_matcher is not None:
            name = var_name_matcher.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for curr_var in init_vars:
        (name, var) = (curr_var[0], curr_var[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names


def default_hparams():
    """
    Default hyper-parameters for GPT-117M
    :return: a HParams set to GPT-117M
    """
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )


def shape_list(x):
    # pylint: disable=invalid-name
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def softmax(x, axis=-1):
    # pylint: disable=invalid-name
    """Modified softmax that does the max trick to prevent large exponent issue"""
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)


def gelu(x):
    # pylint: disable=invalid-name
    """
    GELU activation, described in paper "Gaussian Error Linear Units (GELUs)"
    https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def norm(x, scope, *, axis=-1, epsilon=1e-5):
    # pylint: disable=invalid-name
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x * g + b
        return x


def split_states(x, n):
    # pylint: disable=invalid-name
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m // n])


def merge_states(x):
    # pylint: disable=invalid-name
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a * b])


def dropout(x, dropout_prob):
    # pylint: disable=invalid-name
    """Perform dropout.

    Args:
      x: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return x

    output = tf.nn.dropout(x, 1.0 - dropout_prob)
    return output


def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    # pylint: disable=invalid-name
    """1d convolution i.e., pointwise feed-forward network """
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf],
                            initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b,
                       start + [nf])
        return c


def attention_mask(nd, ns, *, dtype):
    # pylint: disable=invalid-name
    """1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    # pylint: disable=invalid-name
    """attention layer"""
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]
        assert past.shape.ndims == 5

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence],
        # where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(1e10, w.dtype) * (1 - b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        w = dropout(w, hparams.dropout_rate)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state * 3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        a = dropout(a, hparams.dropout_rate)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    # pylint: disable=invalid-name
    """ the point-wise feedforward sub-layer """
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        h2 = dropout(h2, hparams.dropout_rate)
        return h2


def block(x, scope, *, past, hparams):
    # pylint: disable=invalid-name
    """ Single transformer block """
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx * 4, hparams=hparams)
        x = x + m
        return x, present


def past_shape(*, hparams, batch_size=None, sequence=None):
    """ return the shape of the past encoding layers"""
    return [batch_size, hparams.n_layer, 2, hparams.n_head,
            sequence, hparams.n_embd // hparams.n_head]


def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1] * ndims)


def positions_for(tokens, past_length):
    """ generates the position embedding for the token encoding"""
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def model(hparams, input_tokens, past=None, scope='model', reuse=False):
    """
    Default model for LM with tied out weights output
    """
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(input_tokens)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        hidden_layer = tf.gather(wte, input_tokens) + \
                       tf.gather(wpe, positions_for(input_tokens, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for curr_layer, curr_past in enumerate(pasts):
            hidden_layer, present = block(hidden_layer,
                                          'h%d' % curr_layer,
                                          past=curr_past,
                                          hparams=hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        hidden_layer = norm(hidden_layer, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(hidden_layer, [batch * sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results


def classifier_model(hparams, input_tokens, labels, past=None, scope='model', reuse=False):
    """
    Classifier against 6 binary toxicity types (Toxic Comments)
    Applies global max pooling on the base transformer decoder output and
    then dense(6) against that
    """
    with tf.variable_scope(scope, reuse=reuse):
        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.02), )
        past_length = 0 if past is None else tf.shape(past)[-2]
        hidden_layer = tf.gather(wte, input_tokens) + \
                       tf.gather(wpe, positions_for(input_tokens, past_length))

        # Transformer
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for curr_layer, curr_past in enumerate(pasts):
            hidden_layer, _ = block(hidden_layer,
                                    'h%d' % curr_layer,
                                    past=curr_past,
                                    hparams=hparams)
        hidden_layer = norm(hidden_layer, 'ln_f')

        # Classifier model loss.
        # Option 1: Dense All
        # h_flat = tf.reshape(h, [batch, sequence * hparams.n_embd])

        # Option 2: Max pool so that
        # [batch_size, max_seq_len, hidden_size]->[batch_size, hidden_size]
        hidden_layer = tf.keras.layers.GlobalMaxPool1D()(hidden_layer)
        output_weights = tf.get_variable(
            "output_weights", [6, hidden_layer.shape[-1].value],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        logits = tf.matmul(hidden_layer, output_weights, transpose_b=True)
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
        probabilities = tf.nn.sigmoid(logits)
        return loss, probabilities
