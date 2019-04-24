import os
from typing import Tuple
from bigru_fasttext_mlm_auxiliary import BiGRUModellerWithMLM
from custom_losses import MaskedPenalizedSparseCategoricalCrossentropy
from tensorflow.python.keras import layers, losses, optimizers
from tensorflow.python.keras.models import Model
from custom_layers import ReusableEmbedding, TiedOutputEmbedding

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF debug messages

CONFIDENCE_PENALTY = 0.1  # used by MLM loss to penalize overconfident guesses


class BiGRUModellerWithMLMTiedWeights(BiGRUModellerWithMLM):
    def build_bigru_model(self, embedding_matrix) -> Tuple[Model, Model]:
        """
        build and return multi-headed BiGru model
        with 1) MLM output from first GRU layer
             2) standard toxicity classification output from second
        :param embedding_matrix:
        :return:
        """
        token_input = layers.Input(shape=(self.max_seq_len,))
        embedding_layer = ReusableEmbedding(self.vocab_size + 1,
                                            self.embedding_dims,)
        embedded_input, k_embedding_matrix = embedding_layer(token_input)
        lstm1_output = layers.Bidirectional(layers.CuDNNGRU(self.num_neurons,
                                                            return_sequences=True))(embedded_input)
        tied_output = TiedOutputEmbedding()([lstm1_output, k_embedding_matrix])
        aux_output = layers.Activation('softmax', name='aux_output')(tied_output)
        lstm2_output = layers.Bidirectional(layers.CuDNNGRU(self.num_neurons))(lstm1_output)
        main_output = layers.Dense(6, activation='sigmoid', name='main_output')(lstm2_output)

        training_model = Model(inputs=token_input, outputs=[main_output, aux_output])
        training_model.compile(optimizer=optimizers.Adam(),
                               loss={'main_output': losses.binary_crossentropy,
                                     'aux_output': MaskedPenalizedSparseCategoricalCrossentropy(CONFIDENCE_PENALTY)})

        inference_model = Model(inputs=token_input, outputs=main_output)

        print('generated bigru model...')
        print(training_model.summary())

        return training_model, inference_model


if __name__ == '__main__':
    BiGRUModellerWithMLMTiedWeights().run_end_to_end()