"""
Modified version of nvidia-bert for toxic comments specifically.
Also modified over tf_bert to use knowledge distillation
i.e., train against average of loss w/ hard and loss w/ soft targets

Currently using simple average of 4 models for soft targets
1. BERT
2. GPT
3. FastText BiGRU
4. ELMo BiGRU
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import custom_optimization
from tf_transformer import modeling
from tf_transformer import tokenization

SEED = 1337  # seed for generating cross-folds
NUM_FOLDS = 4
TARGET_COLS = ['toxic',
               'severe_toxic',
               'obscene',
               'threat',
               'insult',
               'identity_hate']


flags = tf.flags
FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 2000,
                     "How often to save the model checkpoint.")

flags.DEFINE_bool("use_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU.")

flags.DEFINE_bool("use_xla", False, "Whether to enable XLA JIT compilation.")


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, hard_label=None, soft_label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          hard_label: (Optional) The label of the example. This should be
            specified for train and dev examples, but not for test examples.
          soft_label: Soft targets from ensembled predictions

        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.hard_label = hard_label
        self.soft_label = soft_label


class InputFeatures:
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 hard_label_id,
                 soft_label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.hard_label_id = hard_label_id
        self.soft_label_id = soft_label_id


class ToxicProcessor:
    """Processor for the Toxic data set (version 1)."""
    def __init__(self, seed, data_dir):
        seeded_kf = KFold(n_splits=NUM_FOLDS, random_state=seed, shuffle=True)
        self.train_path = os.path.join(data_dir, "train_with_soft_targets.csv")
        self.test_path = os.path.join(data_dir, "test.csv")
        self.generated_folds = [(train_index, val_index) for train_index, val_index in
                                seeded_kf.split(range(len(pd.read_csv(self.train_path))))]

    def get_train_examples(self, curr_fold):
        """returns training examples for specified fold"""
        return self._create_examples("train", curr_fold)

    def get_val_examples(self, curr_fold):
        """returns validation examples for specified fold"""
        return self._create_examples("val", curr_fold)

    def get_test_examples(self):
        """returns test examples for specified fold"""
        return self._create_examples("test", None)

    def _create_examples(self, mode, curr_fold):
        """
        Creates examples for the data sets.
        Convert toxic comments CSVs into InputExamples
        """
        def get_row_labels(mode, row, return_hard_targets=True):
            if mode == "test":
                return 0, 0, 0, 0, 0, 0

            if return_hard_targets:
                return row.toxic, \
                       row.severe_toxic, \
                       row.obscene, \
                       row.threat, \
                       row.insult, \
                       row.identity_hate
            else:
                return row.soft_toxic, \
                       row.soft_severe_toxic, \
                       row.soft_obscene, \
                       row.soft_threat, \
                       row.soft_insult, \
                       row.soft_identity_hate

        input_file = self.test_path if mode == "test" else self.train_path
        raw_toxic_df = pd.read_csv(input_file)
        if mode == "train":
            raw_toxic_df = raw_toxic_df.iloc[self.generated_folds[curr_fold][0]]
        elif mode == "val":
            raw_toxic_df = raw_toxic_df.iloc[self.generated_folds[curr_fold][1]]
        return [InputExample(guid=row.Index,
                             text_a=row.comment_text,
                             text_b=None,
                             hard_label=get_row_labels(mode, row, return_hard_targets=True),
                             soft_label=get_row_labels(mode, row, return_hard_targets=False)) for row in
                raw_toxic_df.itertuples()]


def convert_single_example(ex_index, example, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    hard_label_id = example.hard_label
    soft_label_id = example.soft_label
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % example.guid)
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("hard label: {}".format(hard_label_id))
        tf.logging.info("soft label: {}".format(soft_label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        hard_label_id=hard_label_id,
        soft_label_id=soft_label_id)
    return feature


def file_based_convert_examples_to_features(
        examples, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            created_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return created_feature

        def create_float_feature(values):
            created_feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return created_feature

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["hard_label_ids"] = create_int_feature(feature.hard_label_id)
        features["soft_label_ids"] = create_float_feature(feature.soft_label_id)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, batch_size):
    """Creates an `input_fn` closure to be passed to Estimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "hard_label_ids": tf.FixedLenFeature([6], tf.int64),
        "soft_label_ids": tf.FixedLenFeature([6], tf.float32)
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            curr_feature = example[name]
            if curr_feature.dtype == tf.int64:
                curr_feature = tf.to_int32(curr_feature)
            example[name] = curr_feature

        return example

    def input_fn(params): # pylint: disable=unused-argument
        """The actual input function."""

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        dataset = tf.data.TFRecordDataset(input_file)
        if is_training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=100000)

        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return dataset

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 hard_labels, soft_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        compute_type=tf.float16 if FLAGS.use_fp16 else tf.float32)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [6, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [6], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        loss = tf.losses.sigmoid_cross_entropy(hard_labels, logits) + \
               tf.losses.sigmoid_cross_entropy(soft_labels, logits)
        probabilities = tf.nn.sigmoid(logits)

        return loss, probabilities


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for Estimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        # pylint: disable=no-member
        """The `model_fn` for Estimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        hard_label_ids = features["hard_label_ids"]
        soft_label_ids = features["soft_label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, hard_label_ids, soft_label_ids,
            use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

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
    """Runs through multi-fold training and eval"""
    # pylint: disable=no-member
    tf.logging.set_verbosity(tf.logging.INFO)
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    fold_auc = 0.
    fold_val_predictions = []

    for curr_fold in range(NUM_FOLDS):
        output_dir = '/tmp/toxic_output/fold{}'.format(curr_fold)
        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

        if FLAGS.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (FLAGS.max_seq_length, bert_config.max_position_embeddings))

        tf.gfile.MakeDirs(output_dir)

        processor = ToxicProcessor(SEED, FLAGS.data_dir)

        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if FLAGS.use_xla:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        run_config = tf.estimator.RunConfig(
            model_dir=output_dir,
            session_config=config,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            keep_checkpoint_max=3)

        train_examples = None
        num_train_steps = None
        num_warmup_steps = None
        if FLAGS.do_train:
            train_examples = processor.get_train_examples(curr_fold)
            num_train_steps = int(
                len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
            num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_one_hot_embeddings=False)

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

        if FLAGS.do_train:
            train_file = os.path.join(output_dir, "train.tf_record")
            file_based_convert_examples_to_features(
                train_examples, FLAGS.max_seq_length, tokenizer, train_file)
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Num examples = %d", len(train_examples))
            tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
            tf.logging.info("  Num steps = %d", num_train_steps)
            train_input_fn = file_based_input_fn_builder(
                input_file=train_file,
                seq_length=FLAGS.max_seq_length,
                is_training=True,
                drop_remainder=True,
                batch_size=FLAGS.train_batch_size)
            estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        if FLAGS.do_eval:
            eval_examples = processor.get_val_examples(curr_fold)
            y_val = np.array([x.hard_label for x in eval_examples])
            y_ids = np.array([x.guid for x in eval_examples])
            num_actual_eval_examples = len(eval_examples)
            eval_file = os.path.join(output_dir, "eval.tf_record")
            file_based_convert_examples_to_features(
                eval_examples, FLAGS.max_seq_length, tokenizer, eval_file)

            tf.logging.info("***** Running evaluation *****")
            tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                            len(eval_examples), num_actual_eval_examples,
                            len(eval_examples) - num_actual_eval_examples)
            tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

            eval_input_fn = file_based_input_fn_builder(
                input_file=eval_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=False,
                batch_size=FLAGS.eval_batch_size)

            val_predictions = estimator.predict(input_fn=eval_input_fn)
            val_prob = np.array([x['probabilities'] for x in val_predictions])
            val_roc_auc_score = roc_auc_score(y_val, val_prob)
            print('ROC-AUC val score: {0:.4f}'.format(val_roc_auc_score))
            fold_auc += val_roc_auc_score
            pred_df = pd.DataFrame(val_prob, columns=TARGET_COLS, index=y_ids)
            fold_val_predictions.append(pred_df)

        if FLAGS.do_predict:
            predict_examples = processor.get_test_examples()
            num_actual_predict_examples = len(predict_examples)
            predict_file = os.path.join(output_dir, "predict.tf_record")
            file_based_convert_examples_to_features(predict_examples,
                                                    FLAGS.max_seq_length, tokenizer,
                                                    predict_file)

            tf.logging.info("***** Running prediction*****")
            tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                            len(predict_examples), num_actual_predict_examples,
                            len(predict_examples) - num_actual_predict_examples)
            tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

            predict_input_fn = file_based_input_fn_builder(
                input_file=predict_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=False,
                batch_size=FLAGS.predict_batch_size)

            result = estimator.predict(input_fn=predict_input_fn)

            output_predict_file = os.path.join(output_dir, "test_results.tsv")
            with tf.gfile.GFile(output_predict_file, "w") as writer:
                num_written_lines = 0
                tf.logging.info("***** Predict results *****")
                for (i, prediction) in enumerate(result):
                    probabilities = prediction["probabilities"]
                    if i >= num_actual_predict_examples:
                        break
                    output_line = "\t".join(
                        str(class_probability)
                        for class_probability in probabilities) + "\n"
                    writer.write(output_line)
                    num_written_lines += 1
            assert num_written_lines == num_actual_predict_examples

        tf.reset_default_graph()

    print('Mean ROC-AUC val score: {0:.4f}'.format(fold_auc / NUM_FOLDS))
    pred_df = pd.concat(fold_val_predictions)
    raw_train_df = pd.read_csv('data/train_with_soft_targets.csv')
    pred_df['id'] = raw_train_df['id'].iloc[pred_df.index.values]
    pred_df = pred_df[['id'] + TARGET_COLS]
    pred_df.to_csv('data/preds_bert_hardsoft_targets.csv', index=False)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    tf.app.run()
