# Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
import operator
import math
from random import random
from collections import defaultdict
from functools import reduce

def sum_probs(probs):
    result = 0
    for _, d in probs.items():
        for _, p in d.items():
            result += p
    return result

def get_ngrams(w, N):
    assert len(w) >= N
    return [w[i:i+N] for i in range(len(w)-N)]


def prob_w(w, N, probs):
    ngrams = get_ngrams(w, N)
    # convert each ngram to its corresponding probability
    w_probs = [probs[ng[:2]][ng[2]] for ng in ngrams]
    return reduce(operator.mul, w_probs)


def entropy(w, N, probs):
    return -1/N * math.log(prob_w(w, N, probs), 2)


def perplexity(w, N, probs):
    return prob_w(w, N, probs) ** (-1/N)