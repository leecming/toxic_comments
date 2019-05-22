"""
Modification of bigru_fasttext_mlm_auxiliary -
Train on val + test set for MLM task and mask out toxicity predictions
"""
import os
from typing import Tuple
import numpy as np
import pandas as pd
# pylint: disable=no-name-in-module
from tensorflow.python.keras import layers, optimizers
from tensorflow.python.keras.models import Model
from sklearn.metrics import log_loss, roc_auc_score
from keras_bigru_fasttext_base import BiGRUBaseModeller
from data_generators import MLMBatchGenerator
from custom_losses import MaskedBinaryCrossedentropy, MaskedPenalizedSparseCategoricalCrossentropy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF debug messages

CONFIDENCE_PENALTY = 0.1  # used by MLM loss to penalize overconfident guesses


class BiGRUModellerWithMLM(BiGRUBaseModeller):
    """ Applies MLM against test labels while ignoring classification outputs"""
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
        mlm_loss = MaskedPenalizedSparseCategoricalCrossentropy(CONFIDENCE_PENALTY)
        training_model.compile(optimizer=optimizers.Adam(),
                               loss={'main_output': MaskedBinaryCrossedentropy(),
                                     'aux_output': mlm_loss})

        inference_model = Model(inputs=token_input, outputs=main_output)

        print('generated bigru model...')
        print(training_model.summary())

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
        # take both train and val for fold
        x_train = train_sequences
        y_train_targets = self.raw_train_df[self.target_cols].values
        y_train = np.zeros((y_train_targets.shape[0],
                            6,
                            2))
        y_train[:, :, 0] = y_train_targets
        y_train[train_indices, :, 1] = 1

        x_train = np.concatenate((x_train, test_sequences))
        y_test = np.zeros((test_sequences.shape[0],
                           6,
                           2))
        y_train = np.concatenate((y_train, y_test))
        print('COMBINED x train shape: {}'.format(x_train.shape))
        print('COMBINED y train shape: {}'.format(y_train.shape))

        train_generator = MLMBatchGenerator(x_train,
                                            y_train,
                                            self.batch_size,
                                            self.vocab_size).batch_generator()

        x_val = train_sequences[val_indices]
        y_val_targets = self.raw_train_df[self.target_cols].iloc[val_indices].values
        y_val = np.ones((y_val_targets.shape[0],
                         6,
                         2))
        y_val[:, :, 0] = y_val_targets
        val_generator = MLMBatchGenerator(x_val,
                                          y_val,
                                          self.batch_size,
                                          self.vocab_size).batch_generator()

        training_model, inference_model = models

        training_model.fit_generator(train_generator,
                                     steps_per_epoch=x_train.shape[0]//self.batch_size,
                                     epochs=self.epochs,
                                     validation_data=val_generator,
                                     validation_steps=len(val_indices)//self.batch_size)

        val_pred = inference_model.predict(x_val,
                                           batch_size=self.batch_size,
                                           verbose=0)
        val_roc_auc_score = roc_auc_score(y_val_targets, val_pred)
        val_log_loss = log_loss(y_val_targets, val_pred)
        print('ROC-AUC val score: {0:.4f}'.format(val_roc_auc_score))
        print('log-loss val score: {0:.4f}'.format(val_log_loss))

        test_predictions = inference_model.predict(test_sequences,
                                                   batch_size=self.batch_size,
                                                   verbose=0)

        return val_roc_auc_score, val_log_loss, test_predictions

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
        fold_log_loss_scores = []
        fold_predictions = []
        for i in range(self.num_folds):
            built_model = self.build_bigru_model(pretrained_embedding_matrix)
            curr_fold_results = self.fit_model_on_fold(built_model, kfold_indices[i],
                                                       k_train_sequences, k_test_sequences)
            fold_roc_auc_scores.append(curr_fold_results[0])
            fold_log_loss_scores.append(curr_fold_results[1])
            fold_predictions.append(curr_fold_results[2])
        print('mean val AUC: {0:.4f}'.format(np.mean(fold_roc_auc_scores)))
        print('mean val log-loss: {0:.4f}'.format(np.mean(fold_log_loss_scores)))
        mean_predictions_df = pd.DataFrame(np.mean(fold_predictions, axis=0),
                                           columns=self.target_cols)
        predicted_test = pd.concat([self.raw_test_df, mean_predictions_df], axis=1)
        predicted_test.to_csv(self.save_predict_path)


if __name__ == '__main__':
    BiGRUModellerWithMLM().run_end_to_end()
