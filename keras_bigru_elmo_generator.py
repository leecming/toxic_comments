"""
Test code to use a generator to pull TF-hub data more quickly
for ELMo
- not useable
"""
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
# pylint: disable=no-name-in-module
# noinspection PyPep8Naming
from tensorflow.python.keras import layers, optimizers, losses
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.text import text_to_word_sequence
from data_generators import ELMoTFHubGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF debug messages


class BiGRUElmoGeneratorModeller:
    """generator based variant of ELMO base"""
    def __init__(self):
        self.seed = 1337
        self.num_folds = 4
        self.batch_size = 128
        self.epochs = 4
        self.num_neurons = 128
        self.max_seq_len = 100
        self.embedding_dims = 1024
        self.num_processes = 1
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
        Use keras tokenizer set to defaults & specified vocab size to
        tokenize the training and test comments
        Then apply pre-padding with val 0.
        :return: tuple of keras Tokenizer and the train & test token sequences
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
        embedded_input = layers.Input(shape=(None, self.embedding_dims))
        gru_output = layers.Bidirectional(layers.CuDNNGRU(self.num_neurons,
                                                          return_sequences=True))(embedded_input)
        gru_output = layers.Bidirectional(layers.CuDNNGRU(self.num_neurons))(gru_output)
        dense_output = layers.Dense(6, activation='sigmoid')(gru_output)

        bigru_model = Model(embedded_input, dense_output)
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
        train_generator = ELMoTFHubGenerator(x_train,
                                             y_train,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             num_processes=self.num_processes).batch_generator()

        x_val = train_sequences[val_indices]
        y_val = self.raw_train_df[self.target_cols].iloc[val_indices].values
        val_generator = ELMoTFHubGenerator(x_val,
                                           y_val,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_processes=self.num_processes).batch_generator()

        test_generator = ELMoTFHubGenerator(test_sequences,
                                            None,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            num_processes=self.num_processes).batch_generator()

        compiled_model.fit(train_generator,
                           steps_per_epoch=len(train_indices) // self.batch_size,
                           epochs=self.epochs,
                           validation_data=val_generator,
                           validation_steps=len(val_indices) // self.batch_size)

        val_pred = compiled_model.predict_generator(val_generator,
                                                    steps=len(val_indices) // self.batch_size,
                                                    verbose=0)
        val_roc_auc_score = roc_auc_score(y_val, val_pred)
        print('ROC-AUC val score: {0:.4f}'.format(val_roc_auc_score))

        test_pred = compiled_model.predict_generator(test_generator,
                                                     steps=len(test_sequences) // self.batch_size,
                                                     verbose=0)

        return val_roc_auc_score, test_pred

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
        mean_predictions_df = pd.DataFrame(np.mean(fold_predictions, axis=0),
                                           columns=self.target_cols)
        predicted_test = pd.concat([self.raw_test_df, mean_predictions_df], axis=1)
        predicted_test.to_csv(self.save_predict_path)


if __name__ == '__main__':
    BiGRUElmoGeneratorModeller().run_end_to_end()
