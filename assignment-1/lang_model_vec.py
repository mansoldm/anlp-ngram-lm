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
char_to_index = {c: i for i, c in enumerate(charset)}


def generate_from_LM_vec(num_to_generate, probs, n):
    chars = '##'
    other = ''.join([charset[randint(0, num_chars-1)] for i in range(n-3)])
    chars += other
    gen_lst = list(chars)

    # 'sliding window': indices will contain the context 
    # i.e. the indices of the n-1 most recent chars
    indices = [char_to_index[c] for c in chars]
    for _ in range(num_to_generate):

        # iterate through each dimension down to the dimension of continuations
        probs_contin = probs
        for i in range(min(len(indices), n-1)):
            ind = indices[i]
            probs_contin = probs_contin[ind]

        # get array of 'continuations' and keep original indices
        probs_contin = list(enumerate(probs_contin))

        if probs_contin.count(probs_contin[0]) == len(probs_contin):
            print("all equal")

        # sort ascendingly by probability
        sorted_tuples = sorted(probs_contin, key=lambda x: x[1])

        max_p = sorted_tuples[-1][1]

        j = len(sorted_tuples)
        for k in reversed(range(len(sorted_tuples))):
            if sorted_tuples[k][1] == max_p:
                j -= 1
            else:
                break

        if j == len(sorted_tuples) - 1:
            j -= np.random.randint(0, 5)

        rdint = randint(j, len(sorted_tuples) - 1)
        max_i = sorted_tuples[rdint][0]
        max_char = charset[max_i]

        if max_char == '#':
            max_char = '\n'
        gen_lst.append(str(max_char))

        # slide window forward
        indices.append(max_i)
        indices = indices[1:]

    # make string and omit the starting ##
    return ''.join(gen_lst)[2:]


def add_alpha_vec(ngram_is, alpha, n):
    probs = np.zeros((num_chars,)*n)

    for indices in ngram_is:
        p = probs
        for i in indices[:-1]:
            p = p[i]
        p[indices[-1]] += 1

    N = np.sum(probs, axis=n-1)
    probs = probs + alpha
    den = N + (alpha * num_chars)
    den_m = np.stack([den for _ in range(num_chars)], axis=n-1)
    probs = probs / den_m
    return probs


def train_model(train, val, alpha_range, n):
    # initialise to 'infinity'
    opt_perp, opt_alpha = float('inf'), float('inf')
    opt_probs = []

    val_f = data_processing.to_string(val)
    val_ngrams = data_processing.get_ngrams(val_f, n)
    val_ngram_is = data_processing.ngrams_to_indices(val_ngrams, char_to_index)

    train_f = data_processing.to_string(train)
    train_ngrams = data_processing.get_ngrams(train_f, n)
    train_ngram_is = data_processing.ngrams_to_indices(train_ngrams, char_to_index)

    for alpha in alpha_range:
        # probs is a 3d matrix of probabilities
        probs = add_alpha_vec(train_ngram_is, alpha, n)

        val_perplexity = data_processing.perplexity_vec(val_ngram_is, probs)
        train_perplexity = data_processing.perplexity_vec(
            train_ngram_is, probs)
        print('alpha: {}, val_perplexity: {}, train_perplexity: {}'.format(
            alpha, val_perplexity, train_perplexity))

        if opt_perp > val_perplexity:
            opt_perp = val_perplexity
            opt_alpha = alpha
            opt_probs = probs

    return opt_probs, opt_alpha
