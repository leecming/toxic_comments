"""
Modification of bigru_fasttext_base
by adding an auxiliary output from first GRU layer to learn
against a masked language model
    1. uses custom loss fn MaskedPenalizedSparseCategoricalCrossentropy
       - a keras SparseCategoricalCrossentropy loss modded to only score masked items
    2. target for auxiliary output shaped as (BATCH_SIZE, MAX_SEQ_LEN, 2)
       - [:, :, 0] = original token ids
       - [:, :, 1] = binary representing whether that token was masked
    3. input token stream modified (at the sample level)
       - p_1 < 0.85 - token not modified
       - p_1 > 0.85
             - p_2 < 0.8 - replace with mask id (first index higher than vocab size)
             - p_2 < 0.9 - replace with random token (1 to vocab size)
             - else - leave word as is (different from p_1 < 0.85 case - still masked)
    4. try mask ID of 0
"""
import os
from typing import Tuple
from bigru_fasttext_base import BiGRUBaseModeller
from utils import MLMBatchGenerator
from custom_losses import MaskedPenalizedSparseCategoricalCrossentropy
from tensorflow.python.keras import layers, losses, optimizers
from tensorflow.python.keras.models import Model
from sklearn.metrics import roc_auc_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF debug messages

CONFIDENCE_PENALTY = 0.1  # used by MLM loss to penalize overconfident guesses


class BiGRUModellerWithMLM(BiGRUBaseModeller):
    def __init__(self):
        super().__init__()
        self.batch_size = 32
        self.vocab_size = 50000
        self.pickled_seq_path = 'data/keras_seq_{}_{}.pkl'.format(self.vocab_size, self.max_seq_len)
        self.pickled_ft_matrix = 'data/ft_matrix_{}.pkl'.format(self.vocab_size)
        self.save_predict_path = 'data/preds_bigru_fasttext_mlm_auxiliary.csv'

    def build_bigru_model(self, embedding_matrix) -> Tuple[Model, Model]:
        """
        build and return multi-headed BiGru model
        with 1) MLM output from first GRU layer
             2) standard toxicity classification output from second
        :param embedding_matrix:
        :return:
        """
        token_input = layers.Input(shape=(self.max_seq_len,))
        embedding_layer = layers.Embedding(self.vocab_size + 1,
                                           self.embedding_dims,
                                           weights=[embedding_matrix],
                                           trainable=False)
        embedded_input = embedding_layer(token_input)
        gru1_output = layers.Bidirectional(layers.CuDNNGRU(self.num_neurons,
                                                           return_sequences=True))(embedded_input)
        aux_output = layers.Dense(self.vocab_size + 1, 'softmax', name='aux_output')(gru1_output)
        gru2_output = layers.Bidirectional(layers.CuDNNGRU(self.num_neurons))(gru1_output)
        main_output = layers.Dense(6, activation='sigmoid', name='main_output')(gru2_output)

        training_model = Model(inputs=token_input, outputs=[main_output, aux_output])
        training_model.compile(optimizer=optimizers.Adam(),
                               loss={'main_output': losses.binary_crossentropy,
                                     'aux_output': MaskedPenalizedSparseCategoricalCrossentropy(CONFIDENCE_PENALTY)})

        inference_model = Model(inputs=token_input, outputs=main_output)

        print('generated bigru model...')
        # print(training_model.summary())

        return training_model, inference_model

    def fit_model_on_fold(self, models: Tuple[Model, Model], curr_fold_indices,
                          train_sequences, test_sequences):
        """
        trains compiled (but previously unfitted) model against given indices
        :param models:
        :param curr_fold_indices:
        :param train_sequences:
        :param test_sequences:
        :return:
        """
        train_indices, val_indices = curr_fold_indices
        x_train = train_sequences[train_indices]
        y_train = self.raw_train_df[self.target_cols].iloc[train_indices].values
        train_generator = MLMBatchGenerator(x_train, y_train, self.batch_size, self.vocab_size).batch_generator()

        x_val = train_sequences[val_indices]
        y_val = self.raw_train_df[self.target_cols].iloc[val_indices].values
        val_generator = MLMBatchGenerator(x_val, y_val, self.batch_size, self.vocab_size).batch_generator()

        training_model, inference_model = models

        training_model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_indices)//self.batch_size,
                                     epochs=self.epochs,
                                     validation_data=val_generator,
                                     validation_steps=len(val_indices)//self.batch_size)

        val_roc_auc_score = roc_auc_score(y_val,
                                          inference_model.predict(x_val,
                                                                  batch_size=self.batch_size, verbose=0))
        print('ROC-AUC val score: {0:.4f}'.format(val_roc_auc_score))

        test_predictions = inference_model.predict(test_sequences, batch_size=self.batch_size, verbose=0)

        return val_roc_auc_score, test_predictions


if __name__ == '__main__':
    BiGRUModellerWithMLM().run_end_to_end()
