"""
Baseline BERT model -
multitask training against MLM and toxic classifications
"""
import os
from typing import Tuple
from keras_bigru_fasttext_base import BiGRUBaseModeller
from data_generators import MLMBatchGenerator
from custom_callbacks import CosineLRSchedule
from custom_losses import MaskedPenalizedSparseCategoricalCrossentropy
import tensorflow as tf
from tensorflow.python.keras import callbacks, losses, optimizers
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from sklearn.metrics import roc_auc_score
from keras_transformer.models import transformer_bert_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF debug messages
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'  # use FP16 to halve memory usage!!!
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1  # JIT compilation
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras


CONFIDENCE_PENALTY = 0.1  # used by MLM loss to penalize overconfident guesses


class BERTBase(BiGRUBaseModeller):
    def __init__(self):
        super().__init__()
        self.batch_size = 96
        self.vocab_size = 50000
        self.learning_rate = 2e-4
        self.embedding_dims = 512
        self.pickled_seq_path = 'data/keras_seq_{}_{}.pkl'.format(self.vocab_size, self.max_seq_len)
        self.pickled_ft_matrix = 'data/ft_matrix_{}.pkl'.format(self.vocab_size)
        self.save_predict_path = 'data/preds_bert_base.csv'

    def generate_embedding_matrix(self, fitted_tokenizer: Tokenizer):
        """
        Not using pre-trained embeddings
        :param fitted_tokenizer:
        :return:
        """
        return None

    def build_bigru_model(self, embedding_matrix) -> Tuple[Model, Model]:
        """
        build and return multi-headed BiGru model
        with 1) MLM output from first GRU layer
             2) standard toxicity classification output from second
        :param embedding_matrix:
        :return:
        """
        training_model, inference_model = transformer_bert_model(
            use_universal_transformer=True,
            max_seq_length=self.max_seq_len,
            vocabulary_size=self.vocab_size + 1,
            word_embedding_size=self.embedding_dims,
            transformer_depth=5,
            num_heads=8)
        training_model.compile(
            optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999),
            loss={'main_output': losses.binary_crossentropy,
                  'aux_output': MaskedPenalizedSparseCategoricalCrossentropy(CONFIDENCE_PENALTY)})

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

        model_callbacks = [callbacks.LearningRateScheduler(
                                CosineLRSchedule(lr_high=self.learning_rate, lr_low=1e-8,
                                                 initial_period=self.epochs),
                           verbose=1)]

        training_model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_indices)//self.batch_size,
                                     epochs=self.epochs,
                                     validation_data=val_generator,
                                     validation_steps=len(val_indices)//self.batch_size,
                                     callbacks=model_callbacks)

        val_roc_auc_score = roc_auc_score(y_val,
                                          inference_model.predict(x_val,
                                                                  batch_size=self.batch_size, verbose=0))
        print('ROC-AUC val score: {0:.4f}'.format(val_roc_auc_score))

        test_predictions = inference_model.predict(test_sequences, batch_size=self.batch_size, verbose=0)

        return val_roc_auc_score, test_predictions


if __name__ == '__main__':
    BERTBase().run_end_to_end()
