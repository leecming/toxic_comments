"""
Babby's first toxic comments classifier
using default keras tokenizer & 2 BiGRUs applied on k-folds train-val split
"""
import os
import pickle
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
# pylint: disable=no-name-in-module
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras import layers, optimizers, losses
from tensorflow.python.keras.models import Model
from fastText import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF debug messages


class BiGRUBaseModeller:
    def __init__(self):
        self.seed = 1337
        self.num_folds = 4
        self.batch_size = 128
        self.epochs = 4
        self.num_neurons = 128
        self.max_seq_len = 100
        self.embedding_dims = 300
        self.vocab_size = 100000
        self.target_cols = ['toxic',
                            'severe_toxic',
                            'obscene',
                            'threat',
                            'insult',
                            'identity_hate']
        self.pickled_seq_path = 'data/keras_seq_{}_{}.pkl'.format(self.vocab_size, self.max_seq_len)
        self.pickled_ft_matrix = 'data/ft_matrix_{}.pkl'.format(self.vocab_size)
        self.save_predict_path = 'data/preds_bigru_fasttext_base.csv'

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

    def texts_to_padded_sequences(self) -> Tuple[Tokenizer, List, List]:
        """
        Use keras tokenizer set to defaults & specified vocab size to
        tokenize the training and test comments
        Then apply pre-padding with val 0.
        :return: tuple of keras Tokenizer and the train & test token sequences
        """
        if os.path.isfile(self.pickled_seq_path):
            with open(self.pickled_seq_path, 'rb') as pickle_file:
                tokenizer, train_sequences, test_sequences = pickle.load(pickle_file)
        else:
            tokenizer = Tokenizer(num_words=self.vocab_size)
            train_test_comment_text = self.raw_train_df['comment_text'].\
                append(self.raw_test_df['comment_text']).reset_index(drop=True)
            tokenizer.fit_on_texts(train_test_comment_text)
            train_sequences = tokenizer.texts_to_sequences(self.raw_train_df['comment_text'])
            train_sequences = pad_sequences(train_sequences, maxlen=self.max_seq_len)
            test_sequences = tokenizer.texts_to_sequences(self.raw_test_df['comment_text'])
            test_sequences = pad_sequences(test_sequences, maxlen=self.max_seq_len)
            with open(self.pickled_seq_path, 'wb') as pickle_file:
                pickle.dump((tokenizer, train_sequences, test_sequences), pickle_file)
        print('generated padded sequences...')
        return tokenizer, train_sequences, test_sequences

    def generate_embedding_matrix(self, fitted_tokenizer: Tokenizer):
        """
        Standard FastText sub-word wikipedia trained model
        :param fitted_tokenizer:
        :return:
        """
        if os.path.isfile(self.pickled_ft_matrix):
            with open(self.pickled_ft_matrix, 'rb') as pickle_file:
                embedding_matrix = pickle.load(pickle_file)
        else:
            ft_model = load_model('data/wiki.en.bin')

            embedding_matrix = np.zeros((self.vocab_size + 1, self.embedding_dims))
            for i in range(1, self.vocab_size + 1):
                try:
                    embedding_matrix[i] = ft_model.get_word_vector(fitted_tokenizer.index_word[i])
                except KeyError:
                    print('FastText OOV?')

            with open(self.pickled_ft_matrix, 'wb') as pickle_file:
                pickle.dump(embedding_matrix, pickle_file)

        print('generated ft embeddings...')

        return embedding_matrix

    def build_bigru_model(self, embedding_matrix) -> Model:
        """
        build and return BiGru model using standard optimizer and loss
        :param embedding_matrix:
        :return:
        """
        token_input = layers.Input(shape=(self.max_seq_len,))
        embedding_layer = layers.Embedding(self.vocab_size + 1,
                                           self.embedding_dims,
                                           weights=[embedding_matrix],
                                           trainable=False)
        embedded_input = embedding_layer(token_input)
        lstm_output = layers.Bidirectional(layers.CuDNNGRU(self.num_neurons,
                                                           return_sequences=True))(embedded_input)
        lstm_output = layers.Bidirectional(layers.CuDNNGRU(self.num_neurons))(lstm_output)
        dense_output = layers.Dense(6, activation='sigmoid')(lstm_output)

        bigru_model = Model(token_input, dense_output)
        bigru_model.compile(optimizer=optimizers.Adam(),
                            loss=losses.binary_crossentropy)

        print('generated bigru model...')

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

        compiled_model.fit(x_train, y_train,
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           validation_data=(x_val, y_val))

        val_roc_auc_score = roc_auc_score(y_val,
                                          compiled_model.predict(x_val,
                                                                 batch_size=self.batch_size, verbose=0))
        print('ROC-AUC val score: {0:.4f}'.format(val_roc_auc_score))

        test_predictions = compiled_model.predict(test_sequences, batch_size=self.batch_size, verbose=0)

        return val_roc_auc_score, test_predictions

    def run_end_to_end(self):
        """
        per the tin, runs text loading, preprocessing and model building and training
        dumps predictions to CSV in same folder
        :return:
        """
        k_tokenizer, k_train_sequences, k_test_sequences = self.texts_to_padded_sequences()
        kfold_indices = self.generate_train_kfolds_indices()
        pretrained_embedding_matrix = self.generate_embedding_matrix(k_tokenizer)

        fold_roc_auc_scores = []
        fold_predictions = []
        for i in range(self.num_folds):
            built_model = self.build_bigru_model(pretrained_embedding_matrix)
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
    BiGRUBaseModeller().run_end_to_end()
