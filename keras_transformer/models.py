"""
Modified version of code from https://github.com/kpot/keras-transformer
1. BERT model modified to hew to toxic comments needs
   - No segmenting/next-sentence objective
   - returns training and inference models separately
"""
# noinspection PyPep8Naming
# pylint: disable=no-name-in-module
from tensorflow.python.keras.layers import Input, Dense, Lambda, Softmax
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import Model
from .extras import ReusableEmbedding, TiedOutputEmbedding
from .position import TransformerCoordinateEmbedding
from .transformer import TransformerACT, TransformerBlock


def transformer_bert_model(
        max_seq_length: int, vocabulary_size: int,
        word_embedding_size: int,
        use_universal_transformer: bool,
        transformer_depth: int,
        num_heads: int,
        transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-4):
    """
    Builds a BERT-based model (Bidirectional Encoder Representations
    from Transformers) following paper "BERT: Pre-training of Deep
    Bidirectional Transformers for Language Understanding"
    (https://arxiv.org/abs/1810.04805)

    Depending on the value passed with `use_universal_transformer` argument,
    this function applies either an Adaptive Universal Transformer (2018)
    or a vanilla Transformer (2017) to do the job (the original paper uses
    vanilla Transformer).
    """
    word_ids = Input(shape=(max_seq_length,), dtype='int32', name='word_ids')
    l2_regularizer = regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None
    embedding_layer = ReusableEmbedding(
        vocabulary_size, word_embedding_size,
        input_length=max_seq_length,
        name='bpe_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer)
    output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='word_prediction_logits')
    output_softmax_layer = Softmax(name='aux_output')
    coordinate_embedding_layer = TransformerCoordinateEmbedding(
        transformer_depth if use_universal_transformer else 1,
        name='coordinate_embedding')

    next_step_input, embedding_matrix = embedding_layer(word_ids)

    if use_universal_transformer:
        # Building a Universal Transformer (2018)
        act_layer = TransformerACT(
            name='adaptive_computation_time')
        transformer_block = TransformerBlock(
            name='transformer', num_heads=num_heads,
            residual_dropout=transformer_dropout,
            attention_dropout=transformer_dropout,
            # Allow bi-directional attention
            use_masking=False)

        act_output = next_step_input
        for i in range(transformer_depth):
            next_step_input = coordinate_embedding_layer(
                next_step_input, step=i)
            next_step_input = transformer_block(next_step_input)
            next_step_input, act_output = act_layer(next_step_input)

        act_layer.finalize()
        next_step_input = act_output
    else:
        # Building a Vanilla Transformer (described in
        # "Attention is all you need", 2017)
        next_step_input = coordinate_embedding_layer(next_step_input, step=0)
        for i in range(transformer_depth):
            next_step_input = (
                TransformerBlock(
                    name='transformer' + str(i), num_heads=num_heads,
                    residual_dropout=transformer_dropout,
                    attention_dropout=transformer_dropout,
                    use_masking=False,  # Allow bi-directional attention
                    vanilla_wiring=True)
                (next_step_input))

    aux_output = output_softmax_layer(
        output_layer([next_step_input, embedding_matrix]))
    cls_node_slice = (
        # selecting the first output position in each sequence
        # (responsible for classification)
        Lambda(lambda x: x[:, 0], name='cls_node_slicer')
        (next_step_input))
    main_output = (
        Dense(6, name='main_output', activation='sigmoid')
        (cls_node_slice))
    training_model = Model(inputs=word_ids, outputs=[main_output, aux_output])
    inference_model = Model(inputs=word_ids, outputs=main_output)
    return training_model, inference_model
