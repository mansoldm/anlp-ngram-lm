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
import const


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


def prob_sequence(ngrams, probs):
    '''Converts ngrams to probabilities'''
    return [probs[ng[:2]][ng[2]] for ng in ngrams]


def log_prob_sequence(ngrams, probs):
    '''Converts ngrams to log probabilities'''
    return [log(probs[ng[:2]][ng[2]], 2) for ng in ngrams]


def entropy(ngrams, probs):
    '''Calculates entropy of given sequence of ngrams'''
    N = len(ngrams)
    log_probs = log_prob_sequence(ngrams, probs)

    return -1/N * sum(log_probs)


def perplexity(ngrams, probs):
    '''Calculates perplexity of given sequence of ngrams'''
    return 2**entropy(ngrams, probs)


##### VECTORISED FUNCTIONS #####
def indices_to_index(indices):
    nc = const.num_chars
    ri = reversed(range(len(indices)))
    return sum([indices[i] * (nc ** i) for i in ri])


def entropy_vec(ngram_is, probs):
    N = len(ngram_is)
    ngram_probs = probs[tuple(ngram_is.T)]
    log_probs = np.array(np.log2(ngram_probs))                
    return -1/N * np.sum(log_probs)


def perplexity_vec(ngram_is, probs):
    return 2**entropy_vec(ngram_is, probs)


def map_to_index(ngram, indices):
    return np.array([indices[c] for c in ngram])


def ngrams_to_indices(ngrams, indices):
    return np.array([map_to_index(ngram, indices) for ngram in ngrams])


def doc_to_ngram_indices(doc, n, indices):
    ngrams = [get_ngrams(line, n) for line in doc]
    # make one flat list
    ngrams_f = [item for line in ngrams for item in line]
    return ngrams_to_indices(ngrams_f, indices)


def map_to_char(indexes, charset):
    return ''.join(charset[i] for i in indexes)


def indices_to_ngrams(mat, charset, num_chars, n):
    lst = list(range(num_chars))
    cartprod = itertools.product(*(lst)*n)
    return [map_to_char(indices, charset) for indices in cartprod]
