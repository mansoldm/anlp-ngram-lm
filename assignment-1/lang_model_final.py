import re
import sys
import re
from random import random, choice, randint
from math import log
from collections import defaultdict
import numpy as np
import data_processing_final
import file_utils_final

charset = ' .0abcdefghijklmnopqrstuvwxyz#'
num_chars = len(charset)
char_to_index = {c: i for i, c in enumerate(charset)}

def generate_from_LM(num_to_generate, probs, n, char_to_index, num_chars, charset):
    # 'sliding window': indices will contain the context
    # i.e. the indices of the n-1 most recent chars
    chars = '#' * (n - 1)
    indices = [char_to_index[c] for c in chars]
    gen_lst = list(chars)

    seen_end, count_gen = False, 0
    while count_gen < num_to_generate:

        # get continuation probs
        probs_contin = probs[tuple(np.transpose(indices))]

        max_i = np.random.choice(range(num_chars), p=probs_contin)
        max_char = charset[max_i]

        gen_lst.append(max_char)
        if max_char != '#' and seen_end:
            gen_lst.append('\n')
            seen_end = False
        elif max_char == '#':
            seen_end = True
        if max_char != '#':
            count_gen += 1

        # slide window forward
        indices.append(max_i)
        indices = indices[1:]

    # make string and omit the starting n-1 #'s
    return ''.join(gen_lst)


def get_perplexity(probs, train_ngram_is, val_ngram_is):
    val_perplexity = data_processing_final.perplexity(val_ngram_is, probs)
    train_perplexity = data_processing_final.perplexity(
        train_ngram_is, probs)

    return train_perplexity, val_perplexity


def add_alpha(counts, alpha, n):
    N = np.sum(counts, axis=n-1)
    probs = counts + alpha
    den = N + (alpha * num_chars)
    den_m = np.stack([den for _ in range(num_chars)], axis=n-1)
    probs = probs / den_m
    return probs


def train_add_alpha(train_ngram_is, val_ngram_is, alpha_range, n, report=True):
    # initialise to 'infinity'
    opt_perp, opt_alpha = float('inf'), float('inf')
    opt_probs = []

    counts = np.zeros((num_chars,)*n)
    ngram_indices, ngram_counts = np.unique(train_ngram_is, return_counts=True, axis=0)
    counts[tuple(ngram_indices.T)] += ngram_counts

    for alpha in alpha_range:
        # probs is a n-d matrix of probabilities
        probs = add_alpha(counts, alpha, n)
        train_perplexity, val_perplexity = get_perplexity(
            probs, train_ngram_is, val_ngram_is)
        if report:
            print('alpha: {}, val_perplexity: {}, train_perplexity: {}'.format(
                alpha, val_perplexity, train_perplexity))

        if opt_perp > val_perplexity:
            opt_perp = val_perplexity
            opt_alpha = alpha
            opt_probs = probs

    return opt_probs, opt_alpha


def gen_interp_lambdas(step, n):
    lambda_perms = data_processing_final.perms(np.arange(0, 1 + 1/step, 1/step), n)
    return [lambdas for lambdas in lambda_perms if sum(lambdas) == 1]


def train_interp(train_ngram_configs, val1_ngram_configs, val2_ngram_is, alpha_range, n):

    # this loop uses the first validation set (val1) to get the optimal alphas
    opt_alphas, add_alpha_probs = [], []
    for i, train_ngram_is, val1_ngram_is in zip(range(n), train_ngram_configs, val1_ngram_configs):
        probs, alpha = train_add_alpha(
            train_ngram_is, val1_ngram_is, alpha_range, i+1, report=False)
        opt_alphas.append(alpha)
        add_alpha_probs.append(probs)

    lambda_configs = gen_interp_lambdas(10, n)

    # this loop uses the second validation set to find the optimal lambda configurations
    opt_perp, opt_lambdas = float('inf'), []
    opt_probs = np.zeros((num_chars,)*n)
    for lambdas in lambda_configs:
        curr_probs = [p for p in add_alpha_probs]
        res_probs = np.zeros((num_chars,)*n)
        for j in range(n):
            res_probs += lambdas[j] * curr_probs[j]

        train_perp, val2_perp = get_perplexity(
            res_probs, train_ngram_is, val2_ngram_is)

        # avoid displaying trailing zeros
        lambdas = [round(l, 3) for l in lambdas]
        print('lambdas: {}, val2_perp: {}, train_perp: {}'.format(
            lambdas, val2_perp, train_perp))

        # if we have obtained a lower perplexity
        if opt_perp > val2_perp:
            opt_perp = val2_perp
            opt_lambdas = lambdas
            opt_probs = res_probs

    return opt_probs, opt_lambdas, opt_alphas
