import re
import sys
import operator
import math
from math import log
from random import random
from collections import defaultdict
from functools import reduce
import itertools
import numpy as np


def to_string(doc):
    '''Converts array of strings to string'''
    return ''.join(doc)


def perms(iterable, length):
    '''Returns permutations of a string'''
    return list(itertools.product(*([iterable]*length)))


def sum_probs(probs):
    '''Returns sum of probabilities of a given model'''
    result = 0
    for _, d in probs.items():
        for _, p in d.items():
            result += p
    return result


def get_ngrams(sequence, n):
    '''Generates sequence of ngrams from string'''
    return [sequence[i:i+n] for i in range(len(sequence)-n)]
    

def entropy(ngram_is, probs):
    N = len(ngram_is)
    ngram_probs = probs[tuple(ngram_is.T)]
    log_probs = np.array(np.log2(ngram_probs))                
    return -1/N * np.sum(log_probs)


def perplexity(ngram_is, probs):
    return 2**entropy(ngram_is, probs)


def map_to_index(ngram, char_to_index):
    return np.array([char_to_index[c] for c in ngram])


def ngrams_to_indices(ngrams, char_to_index):
    return np.array([map_to_index(ngram, char_to_index) for ngram in ngrams])


def doc_to_ngram_indices(doc, n, char_to_index):
    ngrams = [get_ngrams(line, n) for line in doc]
    # make one flat list
    ngrams_f = [item for line in ngrams for item in line]
    return ngrams_to_indices(ngrams_f, char_to_index)


def map_to_char(indexes, charset):
    return ''.join(charset[i] for i in indexes)


def indices_to_ngrams(mat, charset, num_chars, n):
    lst = list(range(num_chars))
    cartprod = itertools.product(*(lst)*n)
    return [map_to_char(indices, charset) for indices in cartprod]
