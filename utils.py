import random
import numpy as np
from itertools import islice

MASK_TOKEN_ID = 0  # try 0 first


class MLMBatchGenerator:
    def __init__(self, data_x, data_y,
                 batch_size, last_token_id):
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.last_token_id = last_token_id

    def batch_generator(self):
        """
        """
        samples = self.generate_samples()
        while True:
            next_bunch_of_samples = islice(samples, self.batch_size)
            mask, sequence, masked_sequence, classifier_targets = zip(*list(next_bunch_of_samples))
            combined_label = np.stack([sequence, mask], axis=-1)
            yield (np.array(masked_sequence), [np.array(classifier_targets), combined_label])

    def generate_samples(self):
        """
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
