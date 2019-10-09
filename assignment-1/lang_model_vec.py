import re
import sys
import re
from random import random, choice, randint
from math import log
from collections import defaultdict
import numpy as np
import data_processing
import file_utils
from const import *

charset = ' .0abcdefghijklmnopqrstuvwxyz#'
num_chars = len(charset)
char_to_index = {c: i for i, c in enumerate(charset)}


def generate_from_LM_vec(num_to_generate, probs, n):
    chars = '#' * (n - 1)
    start_flag = [char_to_index[c] for c in chars]
    gen_lst = list(chars)

    # 'sliding window': indices will contain the context 
    # i.e. the indices of the n-1 most recent chars
    indices = [i for i in start_flag]
    for _ in range(num_to_generate):

        # get continuation probs
        probs_contin = probs[tuple(np.transpose(indices))]

        # get array of 'continuations' and keep original indices
        probs_contin = list(enumerate(probs_contin))

        # sort ascendingly by probability
        sorted_tuples = sorted(probs_contin, key=lambda x: x[1])

        max_p = sorted_tuples[-1][1]

        j = len(sorted_tuples) - 1
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

        gen_lst.append(max_char)

        # slide window forward
        indices.append(max_i)
        indices = indices[1:]

    # make string and omit the starting n-1 #'s
    return ''.join(gen_lst)


def test_perplexity(probs, train_ngram_is, val_ngram_is):
    val_perplexity = data_processing.perplexity_vec(val_ngram_is, probs)
    train_perplexity = data_processing.perplexity_vec(
        train_ngram_is, probs)

    return train_perplexity, val_perplexity


def add_alpha_vec(ngram_is, alpha, n):
    probs = np.zeros((num_chars,)*n)

    # get counts and unique ngram indices
    ngram_indices, counts = np.unique(ngram_is, return_counts=True, axis=0)

    # add each to respective location
    probs[tuple(ngram_indices.T)] += counts

    N = np.sum(probs, axis=n-1)
    probs = probs + alpha
    den = N + (alpha * num_chars)
    den_m = np.stack([den for _ in range(num_chars)], axis=n-1)
    probs = probs / den_m
    return probs


def train_add_alpha(train_ngram_is, val_ngram_is, alpha_range, n):

    # initialise to 'infinity'
    opt_perp, opt_alpha = float('inf'), float('inf')
    opt_probs = []

    for alpha in alpha_range:
        # probs is a 3d matrix of probabilities
        probs = add_alpha_vec(train_ngram_is, alpha, n)
        train_perplexity, val_perplexity = test_perplexity(probs, train_ngram_is, val_ngram_is)
        print('alpha: {}, val_perplexity: {}, train_perplexity: {}'.format(
            alpha, val_perplexity, train_perplexity))

        if opt_perp > val_perplexity:
            opt_perp = val_perplexity
            opt_alpha = alpha
            opt_probs = probs

    return opt_probs, opt_alpha


def gen_interp_lambdas(ind, lambdas, lst, n):
    if ind == n - 1 :
        lambda_n = 1 - sum(lambdas)
        lst.append(lambdas + [lambda_n])
        return
    
    lambda_j = 0 if len(lambdas) == 0 else lambdas[-1]
    for lambda_i in range(0, int(5 - lambda_j * 5)):
        lambda_i /= 5
        gen_interp_lambdas(ind + 1, lambdas + [lambda_i], lst, n)


def train_interp(alpha_range, train, val, n):
    opt_perp, opt_lambdas, opt_probs = float('inf'), [], []
    opt_alpha = float('inf')
    lambda_configs = []
    gen_interp_lambdas(0, [], lambda_configs, n)

    # inds = range(0, n)
    train_ngram_configs = [data_processing.doc_to_ngram_indices(train, i, char_to_index) for i in range(1, n+1)]
    val_ngram_configs = [data_processing.doc_to_ngram_indices(val, i, char_to_index) for i in range(1, n+1)]
    for alpha in alpha_range:
        for lambdas in lambda_configs:
            probs = np.zeros((num_chars,)*n)
            for i, lambda_i in zip(range(0, n), lambdas):
                train_is, val_is = train_ngram_configs[i], val_ngram_configs[i]
                prob_i = add_alpha_vec(train_is, alpha, i+1)
                probs += lambda_i * prob_i

                train_perplexity, val_perplexity = test_perplexity(probs, train_is, val_is)

            print('Alpha: {}, lambdas: {}, val_perp: {}, train_perp: {}'.format(alpha, lambdas, val_perplexity, train_perplexity))
        
            if opt_perp > val_perplexity:
                opt_perp = val_perplexity
                opt_lambdas = lambdas
                opt_probs = probs
                opt_alpha = alpha_range

    return opt_probs, opt_lambdas, opt_alpha


def train_model(train, val, alpha_range, n, estimation_type):

    val_f = data_processing.to_string(val)
    val_ngrams = data_processing.get_ngrams(val_f, n)
    val_ngram_is = data_processing.ngrams_to_indices(val_ngrams, char_to_index)

    train_f = data_processing.to_string(train)
    train_ngrams = data_processing.get_ngrams(train_f, n)
    train_ngram_is = data_processing.ngrams_to_indices(train_ngrams, char_to_index)
    
    if estimation_type == 'add_alpha':
        return train_add_alpha(train_ngram_is, val_ngram_is, alpha_range, n)
    elif estimation_type == 'interpolation':
        return train_interp(alpha_range, train, val, n)