import re
import sys
import re
from random import random, choice, randint
from math import log
from collections import defaultdict
import numpy as np

import data_processing_final
import file_utils_final
from const import charset, num_chars, char_to_index

def entropy(ngram_is, probs):
    N = len(ngram_is)
    ngram_probs = probs[tuple(ngram_is.T)]
    log_probs = np.array(np.log2(ngram_probs))
    return -1/N * np.sum(log_probs)


def perplexity(ngram_is, probs):
    return 2**entropy(ngram_is, probs)


def get_train_val_perplexity(probs, train_ngram_is, val_ngram_is):
    val_perplexity = perplexity(val_ngram_is, probs)
    train_perplexity = perplexity(train_ngram_is, probs)

    return train_perplexity, val_perplexity

def get_perplexity_from_doc(doc, n, probs):
    doc_ngram_is = data_processing_final.doc_to_ngram_indices(
        doc, n, char_to_index)

    return perplexity(doc_ngram_is, probs)


def generate_from_LM(num_to_generate, probs, n):
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



def add_alpha(counts, alpha, n):
    # denominator: sum over all continuations + add num of chars scaled by alpha
    N = np.sum(counts, axis=n-1)
    den = N + (alpha * num_chars)
    den_m = np.stack([den for _ in range(num_chars)], axis=n-1)
    
    # smooth each count in the numerator
    probs = counts + alpha
    probs = probs / den_m
    return probs


def train_add_alpha(train_ngram_is, val_ngram_is, alpha_range, n, report=True):
    opt_perp, opt_alpha = float('inf'), float('inf')
    opt_probs = []

    # get counts of each unique ngram in the training corpus
    counts = np.zeros((num_chars,)*n)
    ngram_indices, ngram_counts = np.unique(
        train_ngram_is, return_counts=True, axis=0)
    counts[tuple(ngram_indices.T)] += ngram_counts

    for alpha in alpha_range:
        # probs is a n-d matrix of probabilities
        probs = add_alpha(counts, alpha, n)
        train_perplexity, val_perplexity = get_train_val_perplexity(
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
    # generate unique lambda sequences which sum to 1
    lambda_perms = data_processing_final.perms(
        np.arange(0, 1 + 1/step, 1/step), n)
    return [lambdas for lambdas in lambda_perms if sum(lambdas) == 1]


def train_interp(train_ngram_configs, val1_ngram_configs, val2_ngram_is, alpha_range, n):
    # get the optimal alphas using the first validation set (val1)
    opt_alphas, add_alpha_probs = [], []
    for i, train_ngram_is, val1_ngram_is in zip(range(n), train_ngram_configs, val1_ngram_configs):
        probs, alpha = train_add_alpha(
            train_ngram_is, val1_ngram_is, alpha_range, i+1, report=False)
        opt_alphas.append(alpha)
        add_alpha_probs.append(probs)

    lambda_configs = gen_interp_lambdas(10, n)

    # get the optimal lambdas using the second validation set (val2)
    opt_perp, opt_lambdas = float('inf'), []
    opt_probs = np.zeros((num_chars,)*n)
    for lambdas in lambda_configs:
        curr_probs = [p for p in add_alpha_probs]
        res_probs = np.zeros((num_chars,)*n)
        for j in range(n):
            res_probs += lambdas[j] * curr_probs[j]

        train_perp, val2_perp = get_train_val_perplexity(
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
