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

def entropy_vec(ngram_is, probs):
    N = len(ngram_is)
    log_probs = - np.log2(probs)
    ngram_probs = np.array([log_probs[i1, i2, i3] for (i1, i2, i3) in ngram_is])
    return -1/N * np.sum(ngram_probs)


def perplexity_vec(ngram_is, probs):
    return 2**entropy_vec(ngram_is, probs)    


def map_to_index(ngram, indices):
    return [indices[c] for c in ngram]


def ngrams_to_indices(ngrams, indices):
    return [map_to_index(ngram, indices) for ngram in ngrams]


def map_to_char(indexes, charset):
    return ''.join(charset[i] for i in indexes)


def indices_to_ngrams(mat, charset):
    shape = np.shape(mat)
    lst = []
    for i in range(len(shape[0])):
        for j in range(len(shape[1])):
            for k in range(len(shape[2])):
                lst.append(map_to_char((i, j, k), charset))
    return lst