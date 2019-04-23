"""
Babby's first toxic comments classifier
using default keras tokenizer & 2 BiGRUs applied on k-folds train-val split
"""
import os
import pickle
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

SEED = 1337
NUM_FOLDS = 4
BATCH_SIZE = 128
EPOCHS = 1
NUM_NEURONS = 128
MAX_SEQ_LEN = 100
EMBEDDING_DIMS = 300
VOCAB_SIZE = 100000
TARGET_COLS = ['toxic',
               'severe_toxic',
               'obscene',
               'threat',
               'insult',
               'identity_hate']
PICKLED_SEQ_PATH = 'data/keras_seq_{}_{}.pkl'.format(VOCAB_SIZE, MAX_SEQ_LEN)

raw_train_df = pd.read_csv('data/train.csv')
raw_test_df = pd.read_csv('data/test.csv')
print('train csv shape: {}'.format(raw_train_df.shape))
print('test csv shape: {}'.format(raw_test_df.shape))
# confirm all 0/1 values
assert all(raw_train_df[TARGET_COLS].apply(lambda x: x.unique() == [0, 1]))


def generate_train_kfolds_indices():
    """
    Seeded kfolds cross validation indices using just a range(len) call
    :return: (training index, validaiton index)-tuple list
    """
    seeded_kf = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
    return [(train_index, val_index) for train_index, val_index in
            seeded_kf.split(range(len(raw_train_df)))]


def texts_to_padded_sequences():
    """
    Use keras tokenizer set to defaults & specified vocab size to
    tokenize the training and test comments
    Then apply pre-padding with val 0.
    :return: tuple of keras Tokenizer and the train & test token sequences
    """
    if os.path.isfile(PICKLED_SEQ_PATH):
        with open(PICKLED_SEQ_PATH, 'rb') as pickle_file:
            tokenizer, train_sequences, test_sequences = pickle.load(pickle_file)
    else:
        tokenizer = Tokenizer(num_words=VOCAB_SIZE)
        train_test_comment_text = raw_train_df['comment_text'].append(raw_test_df['comment_text']).\
            reset_index(drop=True)
        tokenizer.fit_on_texts(train_test_comment_text)
        train_sequences = tokenizer.texts_to_sequences(raw_train_df['comment_text'])
        train_sequences = pad_sequences(train_sequences, maxlen=MAX_SEQ_LEN)
        test_sequences = tokenizer.texts_to_sequences(raw_test_df['comment_text'])
        test_sequences = pad_sequences(test_sequences, maxlen=MAX_SEQ_LEN)
        with open(PICKLED_SEQ_PATH, 'wb') as pickle_file:
            pickle.dump((tokenizer, train_sequences, test_sequences), pickle_file)
    return tokenizer, train_sequences, test_sequences


def generate_embedding_matrix(fitted_tokenizer: Tokenizer):
    """
    Standard FastText sub-word wikipedia trained model
    :param fitted_tokenizer:
    :return:
    """
    ft_model = load_model('data/wiki.en.bin')

    embedding_matrix = np.zeros((VOCAB_SIZE + 1, EMBEDDING_DIMS))
    for i in range(1, VOCAB_SIZE + 1):
        try:
            embedding_matrix[i] = ft_model.get_word_vector(fitted_tokenizer.index_word[i])
        except KeyError:
            print('FastText OOV?')

    return embedding_matrix


def build_bigru_model(embedding_matrix) -> Model:
    """
    build and return BiGru model using standard optimizer and loss
    :param embedding_matrix:
    :return:
    """
    token_input = layers.Input(shape=(MAX_SEQ_LEN,))
    embedding_layer = layers.Embedding(VOCAB_SIZE + 1,
                                       EMBEDDING_DIMS,
                                       weights=[embedding_matrix],
                                       trainable=False)
    embedded_input = embedding_layer(token_input)
    lstm_output = layers.Bidirectional(layers.CuDNNGRU(NUM_NEURONS,
                                                       return_sequences=True))(embedded_input)
    lstm_output = layers.Bidirectional(layers.CuDNNGRU(NUM_NEURONS))(lstm_output)
    dense_output = layers.Dense(6, activation='sigmoid')(lstm_output)

    bigru_model = Model(token_input, dense_output)
    bigru_model.compile(optimizer=optimizers.Adam(),
                        loss=losses.binary_crossentropy)

    return bigru_model


def fit_model_on_fold(compiled_model: Model, curr_fold_indices, train_sequences, test_sequences):
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
    y_train = raw_train_df[TARGET_COLS].iloc[train_indices]
    x_val = train_sequences[val_indices]
    y_val = raw_train_df[TARGET_COLS].iloc[val_indices]

    compiled_model.fit(x_train, y_train,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_data=(x_val, y_val))

    val_roc_auc_score = roc_auc_score(y_val,
                                      compiled_model.predict(x_val,
                                                             batch_size=BATCH_SIZE, verbose=0))
    print('ROC-AUC val score: {0:.4f}'.format(val_roc_auc_score))

    test_predictions = compiled_model.predict(test_sequences, batch_size=BATCH_SIZE, verbose=0)

    return val_roc_auc_score, test_predictions


def run_end_to_end():
    """
    per the tin, runs text loading, preprocessing and model building and training
    dumps predictions to CSV in same folder
    :return:
    """
    k_tokenizer, k_train_sequences, k_test_sequences = texts_to_padded_sequences()
    kfold_indices = generate_train_kfolds_indices()
    pretrained_embedding_matrix = generate_embedding_matrix(k_tokenizer)

    fold_roc_auc_scores = []
    fold_predictions = []
    for i in range(NUM_FOLDS):
        built_model = build_bigru_model(pretrained_embedding_matrix)
        curr_fold_results = fit_model_on_fold(built_model, kfold_indices[i],
                                              k_train_sequences, k_test_sequences)
        fold_roc_auc_scores.append(curr_fold_results[0])
        fold_predictions.append(curr_fold_results[1])
    print('mean val AUC: {0:.4f}'.format(np.mean(fold_roc_auc_scores)))
    mean_predictions_df = pd.DataFrame(np.mean(fold_predictions, axis=0),
                                       columns=TARGET_COLS)
    predicted_test = pd.concat([raw_test_df, mean_predictions_df], axis=1)
    predicted_test.to_csv('./predictions_bigru.csv')


if __name__ == '__main__':
    run_end_to_end()
