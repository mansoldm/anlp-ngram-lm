# Here are some libraries you're likely to use. You might lineant/need others as lineell.
import re
import sys
import operator
import math
from math import log
from random import random
from collections import defaultdict
from functools import reduce
import itertools


def flatten(doc):
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
