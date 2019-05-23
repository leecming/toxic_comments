"""
Used pre-trained ELmo embeddings from tensorflow-hub
tokenize using default keras settings and fix length to max_seq_length
we send the tokens as re-combined string (with tf-hub re-tokenizing on spaces)
makes it easier to invoke the module calls

Architecture goes to BiGRU
"""
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
# pylint: disable=no-name-in-module
# noinspection PyPep8Naming
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers, optimizers, losses
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.text import text_to_word_sequence
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF debug messages
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'  # use FP16 to halve memory usage!!!

ELMO_TF_HUB_URL = 'https://tfhub.dev/google/elmo/2'


def elmo_embedding(raw_text):
    """calls to tf hub to get word embeddings, tokenization happens there"""
    return hub.Module(ELMO_TF_HUB_URL, trainable=False)(tf.squeeze(tf.cast(raw_text, tf.string)),
                                                        signature='default', as_dict=True)['elmo']


class BiGRUElmoBaseModeller:
    """Basic BiGRU used with ELMo TF hub as embedding layer"""
    def __init__(self):
        self.seed = 1337
        self.num_folds = 4
        self.batch_size = 128
        self.epochs = 4
        self.num_neurons = 128
        self.max_seq_len = 100
        self.embedding_dims = 1024
        self.vocab_size = 100000
        self.target_cols = ['toxic',
                            'severe_toxic',
                            'obscene',
                            'threat',
                            'insult',
                            'identity_hate']
        self.save_predict_path = 'data/preds_bigru_elmo_base.csv'

        self.raw_train_df = pd.read_csv('data/train.csv')
        self.raw_test_df = pd.read_csv('data/test.csv')
        print('train csv shape: {}'.format(self.raw_train_df.shape))
        print('test csv shape: {}'.format(self.raw_test_df.shape))
        # confirm all 0/1 values
        assert all(self.raw_train_df[self.target_cols].apply(lambda x: x.unique() == [0, 1]))

    def generate_train_kfolds_indices(self) -> List:
        """
        Seeded kfolds cross validation indices using just a range(len) call
        :return: (training index, validation index)-tuple list
        """
        seeded_kf = KFold(n_splits=self.num_folds, random_state=self.seed, shuffle=True)
        print('generated train kfold indices...')
        return [(train_index, val_index) for train_index, val_index in
                seeded_kf.split(range(len(self.raw_train_df)))]

    def texts_to_padded_sequences(self) -> Tuple[List, List]:
        """
        we're passing tf hub the raw text but need to do
        parsing here to make sure num tokens is fixed to max_seq_len
        """
        def split_and_pad(input_string):
            tokenised_string = text_to_word_sequence(input_string)
            return ' '.join(tokenised_string[:min(self.max_seq_len, len(tokenised_string))])

        train_sequences = self.raw_train_df['comment_text'].apply(split_and_pad).values
        test_sequences = self.raw_test_df['comment_text'].apply(split_and_pad).values

        print('generated padded sequences...')
        return train_sequences, test_sequences

    def build_bigru_model(self) -> Model:
        """
        build and return BiGru model using standard optimizer and loss
        :return:
        """
        token_input = layers.Input(shape=(1,), dtype='string')
        embedded_input = layers.Lambda(elmo_embedding,
                                       output_shape=(self.embedding_dims, ))(token_input)
        gru_output = layers.Bidirectional(layers.CuDNNGRU(self.num_neurons,
                                                          return_sequences=True))(embedded_input)
        gru_output = layers.Bidirectional(layers.CuDNNGRU(self.num_neurons))(gru_output)
        dense_output = layers.Dense(6, activation='sigmoid')(gru_output)

        bigru_model = Model(token_input, dense_output)
        bigru_model.compile(optimizer=optimizers.Adam(),
                            loss=losses.binary_crossentropy)

        print('generated bigru model...')
        print(bigru_model.summary())

        return bigru_model

    def fit_model_on_fold(self, compiled_model: Model, curr_fold_indices,
                          train_sequences, test_sequences):
        """
        trains compiled (but previously unfitted) model against given indices
        :param compiled_model:
        :param curr_fold_indices:
        :param train_sequences:
        :param test_sequences:
        :return:
        """
        train_indices, val_indices = curr_fold_indices
        x_train = train_sequences[train_indices]
        y_train = self.raw_train_df[self.target_cols].iloc[train_indices].values
        x_val = train_sequences[val_indices]
        y_val = self.raw_train_df[self.target_cols].iloc[val_indices].values

        with tf.Session() as session:
            K.set_session(session)
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())

            compiled_model.fit(x_train, y_train,
                               batch_size=self.batch_size,
                               epochs=self.epochs,
                               validation_data=(x_val, y_val))

            val_pred = compiled_model.predict(x_val, batch_size=self.batch_size, verbose=0)
            val_roc_auc_score = roc_auc_score(y_val, val_pred)
            print('ROC-AUC val score: {0:.4f}'.format(val_roc_auc_score))

            val_df = pd.DataFrame(val_pred, index=val_indices)
            val_df.columns = self.target_cols

        return val_roc_auc_score, val_df

    def run_end_to_end(self):
        """
        per the tin, runs text loading, preprocessing and model building and training
        dumps predictions to CSV in same folder
        :return:
        """
        k_train_sequences, k_test_sequences = self.texts_to_padded_sequences()
        kfold_indices = self.generate_train_kfolds_indices()

        fold_roc_auc_scores = []
        fold_predictions = []
        for i in range(self.num_folds):
            built_model = self.build_bigru_model()
            curr_fold_results = self.fit_model_on_fold(built_model, kfold_indices[i],
                                                       k_train_sequences, k_test_sequences)
            fold_roc_auc_scores.append(curr_fold_results[0])
            fold_predictions.append(curr_fold_results[1])
        print('mean val AUC: {0:.4f}'.format(np.mean(fold_roc_auc_scores)))
        pred_df = pd.concat(fold_predictions)
        pred_df['id'] = self.raw_train_df['id'].iloc[pred_df.index.values]
        pred_df = pred_df[['id'] + self.target_cols]
        pred_df.to_csv(self.save_predict_path, index=False)


if __name__ == '__main__':
    config = tf.ConfigProto()
    # pylint:disable=no-member
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    BiGRUElmoBaseModeller().run_end_to_end()
