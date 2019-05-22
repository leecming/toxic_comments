"""
Modified version of code from https://github.com/openai/gpt-2
Byte pair encoding utilities
"""

import os
from functools import lru_cache
import json
import regex as re


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you
    end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    list_ordinal = list(range(ord("!"), ord("~") + 1)) + \
        list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    list_chr = list_ordinal[:]
    ordinal_index = 0
    for num in range(2 ** 8):
        if num not in list_ordinal:
            list_ordinal.append(num)
            list_chr.append(2 ** 8 + ordinal_index)
            ordinal_index += 1
    list_chr = [chr(n) for n in list_chr]
    return dict(zip(list_ordinal, list_chr))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    """
    BPE encoder that loads initialized vocab and bpe rank files,
    and provides helper functions to convert raw text to bpe tokens
    and to convert bpe tokens back to raw text
    """

    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should have added re.IGNORECASE so BPE merges can
        # happen for capitalized versions of contractions
        re_string = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pat = re.compile(re_string)

    def bpe(self, token):
        """
        Given text token, iteratively generates bpe tuple generated
        from bpe rankings file
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """
        converts raw text into stream of bpe tokens
        :param text: raw text string to be encode in bpe tokens
        :return: list of bpe token ids
        """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        """
        converts list of bpe tokens back into raw text
        :param tokens: bpe tokens
        :return: raw text
        """
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


def get_encoder(model_name):
    """
    Helper function to load an encoder from a pre-trained model
    :param model_name: currently either 117M, 345M pre-trained models provided by OpenAI
    :return: an initialized Encoder object
    """
    with open(os.path.join('data', 'models', model_name, 'encoder.json'), 'r') as encoder_json_file:
        encoder = json.load(encoder_json_file)
    with open(os.path.join('data', 'models', model_name, 'vocab.bpe'),
              'r', encoding="utf-8") as vocab_bpe_file:
        bpe_data = vocab_bpe_file.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )
