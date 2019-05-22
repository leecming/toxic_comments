""" Custom data generators"""

import random
import multiprocessing as mp
from itertools import islice, zip_longest
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

MASK_TOKEN_ID = 0  # try 0 first
ELMO_TF_HUB_URL = 'https://tfhub.dev/google/elmo/2'


def chunker(chunked_size, iterable):
    """breaks up the iterable into chunks of size chunked_size"""
    return zip_longest(*[iter(iterable)]*chunked_size)


class ELMoTFHubGenerator:
    """An attempt at wrapping a generator around ELMo TF Hub calls
       Runs on CPU - in practice, since it's running an actual TF graph, too slow to be useable
    """
    def __init__(self, data_x, data_y,
                 batch_size, shuffle=False,
                 num_processes=4):
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_processes = num_processes

    def elmo_producer(self,
                      in_queue: mp.Queue,
                      out_queue: mp.Queue,
                      terminate_event: mp.Event):
        """ Producer called by MP to generate ELMO embeddings from tf hub"""
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        elmo = hub.Module(ELMO_TF_HUB_URL, trainable=False)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member

        with tf.Session(config=config) as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())

            while not terminate_event.is_set():
                batch_idx = list(in_queue.get(True, timeout=2))
                batch_comments = self.data_x[batch_idx]

                batch_embeddings = session.run(elmo(batch_comments,
                                                    signature='default',
                                                    as_dict=True)['elmo'])

                if self.data_y is None:
                    out_queue.put(batch_embeddings)
                else:
                    batch_y = self.data_y[batch_idx]
                    out_queue.put((batch_embeddings, batch_y))

    def batch_generator(self):
        """ creates an MP pool to call ELMO tf hub and generate the embeddings"""
        idx_queue = mp.Queue()
        embeddings_queue = mp.Queue(maxsize=self.num_processes * 2)
        terminate_event = mp.Event()
        pool = mp.Pool(self.num_processes,
                       initializer=self.elmo_producer,
                       initargs=(idx_queue, embeddings_queue, terminate_event))

        try:
            while True:
                idx_list = list(range(len(self.data_x)))
                if self.shuffle:
                    random.shuffle(idx_list)
                batched_idx = list(chunker(self.batch_size, idx_list))
                for batch_idx in batched_idx:
                    idx_queue.put(batch_idx)

                for _ in range(len(batched_idx)):
                    yield embeddings_queue.get()
        finally:
            terminate_event.set()
            pool.close()
            pool.join()


class MLMBatchGenerator:
    """ Creates batches of MLM samples specifically for Toxic"""
    def __init__(self, data_x, data_y,
                 batch_size, last_token_id):
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.last_token_id = last_token_id

    def batch_generator(self):
        """
        yields 1) the masked sequence of token ids as a feature,
               2) tuple of the toxic classifier targets,
                  and the original token id sequence with positions of masking
        """
        samples = self.generate_samples()
        while True:
            next_bunch_of_samples = islice(samples, self.batch_size)
            mask, sequence, masked_sequence, classifier_targets = zip(*list(next_bunch_of_samples))
            combined_label = np.stack([sequence, mask], axis=-1)
            yield (np.array(masked_sequence), [np.array(classifier_targets), combined_label])

    def generate_samples(self):
        """
        Masks the sequence using odds given by the paper - 15% masking,
        of which 80% are masked with with the MASK token,
        10% are flipped to another token
        10% left as the original token
        """
        while True:
            idx_list = list(range(len(self.data_x)))
            random.shuffle(idx_list)

            for curr_idx in idx_list:
                sequence = self.data_x[curr_idx]
                masked_sequence = sequence.copy()
                output_mask = np.zeros((len(sequence),), dtype=int)
                #  since we're applying padding, need to constrain range of masking
                try:
                    start_idx = np.flatnonzero(sequence)[0]
                except IndexError:  # all zeros - due to tokenizing on cut-down vocab
                    continue
                for word_pos in range(start_idx, len(sequence)):
                    if random.random() < 0.15:
                        dice = random.random()
                        if dice < 0.8:  # 80% of 15% = mask
                            masked_sequence[word_pos] = MASK_TOKEN_ID
                        elif dice < 0.9:  # 10% of 15% = random
                            masked_sequence[word_pos] = random.randint(
                                1, self.last_token_id)
                        # else: 10% of 15% just leave the word as is
                        output_mask[word_pos] = 1
                yield (output_mask, sequence, masked_sequence, self.data_y[curr_idx])
