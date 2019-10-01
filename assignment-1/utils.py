#Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
import operator
import math
from random import random
from collections import defaultdict
from functools import reduce


def get_ngrams(w, N) -> list:
    assert len(w) >= N
    return [w[i:i+N] for i in range(len(w)-N)]


def prob_w(w, N, probs):
    n_grams = get_ngrams(w, N)
    # convert each ngram to its corresponding probability
    w_probs = map(lambda ng: probs[ng], n_grams)
    return reduce(operator.mul, w_probs)


def entropy(w, N, probs):
    return -1/N * math.log(prob_w(w, N, probs), 2)


def perplexity(w, N, probs):
    return prob_w(w, N, probs) ** (-1/N)
