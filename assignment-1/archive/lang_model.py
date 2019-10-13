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


def generate_from_LM(N, probs):
    c1, c2 = '#', '#'
    gen_lst = [c1, c2]
    for _ in range(N):
        bigram = str(c1) + str(c2)
        # convert dict of 'continuations' to list of (char, prob) tuples
        print('*' + bigram + '*')
        probs_bigram = probs[bigram].items()

        # sort ascendingly by probability
        # print(probs_bigram)
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


def add_alpha(docs, alpha):
    trigram_counts = defaultdict(float)
    trigram_smoothed = defaultdict(dict)
    bigram_counts = defaultdict(float)
    for doc in docs:
        ngrams = data_processing.get_ngrams(doc, 3)
        bigrams = data_processing.get_ngrams(doc, 2)

        for ngram in ngrams:
            trigram_counts[ngram] += 1

        for bigram in bigrams:
            bigram_counts[bigram] += 1

    # add smoothing
    # trigrams
    alpha_d = alpha * num_chars
    for c1, c2, c3 in perms:
        bigram = c1+c2
        trigram = bigram + c3

        addend = (trigram_counts[trigram] + alpha) / \
            (bigram_counts[bigram] + alpha_d)
        nested_d = trigram_smoothed[bigram]
        if c3 in nested_d:
            nested_d[c3] += addend
        else:
            nested_d[c3] = addend

        trigram_smoothed[bigram] = nested_d

    return trigram_smoothed


def train_model(train, val, alpha_range):
    # initialise to 'infinity'
    opt_perp, opt_alpha = float('inf'), float('inf')
    opt_probs = []
    for alpha in alpha_range:
        probs = add_alpha(train, alpha)

        val_f = data_processing.to_string(val)
        val_ngrams = data_processing.get_ngrams(val_f, 3)

        train_f = data_processing.to_string(train)
        train_ngrams = data_processing.get_ngrams(train_f, 3)

        val_perplexity = data_processing.perplexity(val_ngrams, probs)
        train_perplexity = data_processing.perplexity(train_ngrams, probs)
        print('alpha: {}, val_perplexity: {}, train_perplexity: {}'.format(
            alpha, val_perplexity, train_perplexity))

        if opt_perp > val_perplexity:
            opt_perp = val_perplexity
            opt_alpha = alpha
            opt_probs = probs

    return opt_probs, opt_alpha
