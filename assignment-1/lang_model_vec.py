import re
import sys
import re
from random import random, choice, randint
from math import log
from collections import defaultdict
import numpy as np
import data_processing
import file_utils


charset = ' .0abcdefghijklmnopqrstuvwxyz#'
num_chars = len(charset)
d = num_chars ** 3
indices = {c: i for i, c in enumerate(charset)}

perms = data_processing.perms(list(charset), 3)

def generate_from_LM_vec(N, probs):
    c1, c2 = '#', '#'
    gen_lst = [c1, c2]
    for _ in range(N):
        bigram = str(c1) + str(c2)
        # convert dict of 'continuations' to list of (char, prob) tuples
        probs_bigram = probs[bigram].items()

        # sort ascendingly by probability
        sorted_tuples = sorted(probs_bigram, key=lambda x: x[1])
        max_p = sorted_tuples[-1]

        j = len(sorted_tuples)
        for k in reversed(range(len(sorted_tuples))):
            if sorted_tuples[k] == max_p:
                j -= 1
            else:
                break

        if j == len(sorted_tuples) - 1:
            j -= np.random.randint(0, 5)

        rdint = randint(j, len(sorted_tuples) - 1)
        max_char = sorted_tuples[rdint][0]

        if max_char == '#':
            max_char = '\n'
        gen_lst.append(str(max_char))

        c1 = c2
        c2 = max_char

    return ''.join(gen_lst)[2:]


def add_alpha_vec(docs, alpha):
    probs = np.zeros((num_chars, num_chars, num_chars))
    for doc in docs:
        n_grams = data_processing.get_ngrams(doc, 3)
        for ngram in n_grams:
            i0, i1, i2 = data_processing.map_to_index(ngram, indices)
            probs[i0, i1, i2] += 1

    N = np.sum(probs, axis=2)
    probs = probs + alpha
    den = N + (alpha * num_chars)
    den_m = np.stack([den for _ in range(num_chars)], axis=2)
    probs = probs / den_m
    return probs


def train_model(train, val, alpha_range):
    # initialise to 'infinity'
    opt_perp, opt_alpha = float('inf'), float('inf')
    opt_probs = []

    val_f = data_processing.to_string(val)
    val_ngrams = data_processing.get_ngrams(val_f, 3)
    val_ngram_is = data_processing.ngrams_to_indices(val_ngrams, indices)

    train_f = data_processing.to_string(train)
    train_ngrams = data_processing.get_ngrams(train_f, 3)
    train_ngram_is = data_processing.ngrams_to_indices(train_ngrams, indices)

    for alpha in alpha_range:
        # probs is a 3d matrix of probabilities
        probs = add_alpha_vec(train, alpha)

        val_perplexity = data_processing.perplexity_vec(val_ngram_is, probs)
        train_perplexity = data_processing.perplexity_vec(train_ngram_is, probs)
        print('alpha: {}, val_perplexity: {}, train_perplexity: {}'.format(
            alpha, val_perplexity, train_perplexity))

        if opt_perp > val_perplexity:
            opt_perp = val_perplexity
            opt_alpha = alpha
            opt_probs = probs

    return opt_probs, opt_alpha
