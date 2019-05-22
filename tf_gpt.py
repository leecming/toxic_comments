""" Modified version of gpt for toxic comments"""
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, roc_auc_score
from tf_gpt import encoder
from tf_gpt import model
import custom_optimization

flags = tf.flags
FLAGS = flags.FLAGS

TOXIC_TRAIN_PATH = os.path.join('data', 'train.csv')
TOXIC_TEST_PATH = os.path.join('data', 'test.csv')
TARGET_COLS = ['toxic',
               'severe_toxic',
               'obscene',
               'threat',
               'insult',
               'identity_hate']

flags.DEFINE_integer(
    'batch_size', 32,
    'batch size used for training, eval, and prediction')

flags.DEFINE_integer(
    'num_folds', 4,
    'Number of folds for cross-validation')

flags.DEFINE_integer(
    'seed', 1337,
    'Seed to use for generating fold indices')

flags.DEFINE_string(
    'model_name', '345M',
    'Flag whether to use the pretrained 117M or 345M models')

flags.DEFINE_integer(
    'max_seq_length', 96,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')

flags.DEFINE_float('learning_rate', 5e-5, 'The initial learning rate for Adam.')

flags.DEFINE_float('num_train_epochs', 3.0,
                   'Total number of training epochs to perform.')

flags.DEFINE_float('dropout_rate', 0.,
                   'Dropout rate applied during training')

flags.DEFINE_float(
    'warmup_proportion', 0.1,
    'Proportion of training to perform linear learning rate warmup for. '
    'E.g., 0.1 = 10% of training.')

flags.DEFINE_integer('save_checkpoints_steps', 2000,
                     'How often to save the model checkpoint.')

flags.DEFINE_bool('use_fp16', False, 'Whether to use fp32 or fp16 arithmetic on GPU.')

flags.DEFINE_bool('use_xla', False, 'Whether to enable XLA JIT compilation.')


class GPTToxicClassifier:
    """Wrapper with preprocessor and model builder using custom tf estimator"""
    def __init__(self, model_name: str):
        """
        Initializes encoder from pre-trained model's vocab & bpe merges,
        its hyper-parameters (i.e., Transformer decoder params) &
        loads up the Toxic comments training & test csvs into dataframes
        :param model_name: '117M' or '345M'
        """
        self.enc = encoder.get_encoder(model_name)
        self.hparams = model.default_hparams()
        self.hparams.add_hparam('dropout_rate', 0.)  # add dropout for training
        with open(os.path.join('data',
                               'models',
                               model_name,
                               'hparams.json')) as param_file:
            self.hparams.override_from_dict(json.load(param_file))

        self.train_df = pd.read_csv(TOXIC_TRAIN_PATH)
        self.test_df = pd.read_csv(TOXIC_TEST_PATH)
        self.train_sequences = None
        self.train_labels = None
        self.test_sequences = None

        seeded_kf = KFold(n_splits=FLAGS.num_folds, random_state=FLAGS.seed, shuffle=True)
        self.train_folds = [(train_index, val_index) for train_index, val_index in
                            seeded_kf.split(range(len(self.train_df)))]

    def generate_all_sequences_and_labels(self, generate_test_seq=False):
        """
        Encodes and pads (pad/truncate = pre) up till max_seq_length
        TODO currently defaulting to 0s for padding & not demarcating sentence start/end
        :return:
        """
        list_train_encodings = self.train_df['comment_text'].map(self.enc.encode).values
        self.train_sequences = pad_sequences(list_train_encodings, maxlen=FLAGS.max_seq_length)
        self.train_labels = self.train_df[TARGET_COLS].values
        if generate_test_seq:
            list_test_encodings = self.test_df['comment_text'].map(self.enc.encode).values
            self.test_sequences = pad_sequences(list_test_encodings, maxlen=FLAGS.max_seq_length)

    @staticmethod
    def input_fn_builder(is_training=False):
        """vanilla input function builder using in-memory array"""
        def input_fn(features, labels, batch_size):
            # pylint: disable=unused-argument
            """The actual input function."""
            dataset = tf.data.Dataset.from_tensor_slices(features)
            if is_training:
                dataset = dataset.repeat()
                dataset = dataset.shuffle(buffer_size=100000)
            return dataset.batch(batch_size)
        return input_fn

    def model_fn_builder(self, learning_rate, num_train_steps, num_warmup_steps):
        # pylint: disable=no-member
        """vanilla model builder for custom estimator"""
        def model_fn(features, labels, mode, params):
            # pylint: disable=unused-argument
            input_ids = features['input_ids']
            label_ids = features['label_ids']

            if mode == tf.estimator.ModeKeys.TRAIN:
                self.hparams.dropout_rate = FLAGS.dropout_rate
            else:
                self.hparams.dropout_rate = 0.

            total_loss, probabilities = model.classifier_model(hparams=self.hparams,
                                                               input_tokens=input_ids,
                                                               labels=label_ids,
                                                               past=None,
                                                               reuse=tf.AUTO_REUSE)

            tvars = tf.trainable_variables()
            init_checkpoint = os.path.join('data', 'models', FLAGS.model_name)
            (assignment_map,
             initialized_variable_names) = model.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info('**** Trainable Variables ****')
            for var in tvars:
                init_string = ''
                if var.name in initialized_variable_names:
                    init_string = ', *INIT_FROM_CKPT*'
                tf.logging.info('  name = %s, shape = %s%s', var.name, var.shape, init_string)

            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = custom_optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, False,
                    None, FLAGS.use_fp16)

                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op)
            elif mode == tf.estimator.ModeKeys.EVAL:
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss)
            else:
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={"probabilities": probabilities})
            return output_spec

        return model_fn


def main(_):
    # pylint: disable=no-member
    """Runs through multi fold train eval"""
    tf.logging.set_verbosity(tf.logging.INFO)
    gpt_classifier = GPTToxicClassifier(FLAGS.model_name)
    gpt_classifier.generate_all_sequences_and_labels()

    fold_auc = 0
    fold_log_loss = 0
    train_input_fn = GPTToxicClassifier.input_fn_builder(is_training=True)
    test_input_fn = GPTToxicClassifier.input_fn_builder(is_training=False)

    for curr_fold in range(FLAGS.num_folds):
        # START OF FOLD
        output_dir = '/tmp/toxic_output/fold{}'.format(curr_fold)
        tf.gfile.MakeDirs(output_dir)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if FLAGS.use_xla:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        run_config = tf.estimator.RunConfig(
            model_dir=output_dir,
            session_config=config,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            keep_checkpoint_max=3)

        fold_train_indices, fold_val_indices = gpt_classifier.train_folds[curr_fold]
        fold_x_train = np.take(gpt_classifier.train_sequences, fold_train_indices, axis=0)
        fold_y_train = np.take(gpt_classifier.train_labels, fold_train_indices, axis=0)
        train_features = {'input_ids': fold_x_train,
                          'label_ids': fold_y_train}
        fold_x_val = np.take(gpt_classifier.train_sequences, fold_val_indices, axis=0)
        fold_y_val = np.take(gpt_classifier.train_labels, fold_val_indices, axis=0)
        val_features = {'input_ids': fold_x_val,
                        'label_ids': fold_y_val}

        num_train_steps = int(fold_x_train.shape[0] / FLAGS.batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        model_fn = gpt_classifier.model_fn_builder(learning_rate=FLAGS.learning_rate,
                                                   num_train_steps=num_train_steps,
                                                   num_warmup_steps=num_warmup_steps)

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

        tf.logging.info("***** Running training for fold %d*****", curr_fold)
        tf.logging.info("  Num examples = %d", fold_x_train.shape[0])
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        estimator.train(input_fn=lambda: train_input_fn(train_features,
                                                        None,
                                                        FLAGS.batch_size),
                        max_steps=num_train_steps)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", fold_x_val.shape[0])
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        val_predictions = estimator.predict(input_fn=lambda: test_input_fn(val_features,
                                                                           None,
                                                                           FLAGS.batch_size))
        val_prob = np.array([x['probabilities'] for x in val_predictions])
        val_roc_auc_score = roc_auc_score(fold_y_val, val_prob)
        val_log_loss = log_loss(fold_y_val, val_prob)
        print('ROC-AUC val score: {0:.4f}'.format(val_roc_auc_score))
        print('log loss val score: {0:.4f}'.format(val_log_loss))
        fold_auc += val_roc_auc_score
        fold_log_loss += val_log_loss
        tf.reset_default_graph()
        # END OF FOLD
    print('Mean AUC: {0:.4f}'.format(fold_auc / FLAGS.num_folds))
    print('Mean log-loss: {0:.4f}'.format(fold_log_loss / FLAGS.num_folds))


if __name__ == '__main__':
    tf.app.run()
