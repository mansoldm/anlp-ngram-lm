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

        max_i = np.random.choice(range(num_chars), p=probs_contin)
        max_char = charset[max_i]

        gen_lst.append(max_char)
        if max_char == '#': 
            gen_lst.append('\n')

        # slide window forward
        indices.append(max_i)
        indices = indices[1:]

    # make string and omit the starting n-1 #'s
    return ''.join(gen_lst)


def get_perplexity(probs, train_ngram_is, val_ngram_is):
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


def train_add_alpha(train_ngram_is, val_ngram_is, alpha_range, n, report=True):
    # initialise to 'infinity'
    opt_perp, opt_alpha = float('inf'), float('inf')
    opt_probs = []

    for alpha in alpha_range:
        # probs is a nd matrix of probabilities
        probs = add_alpha_vec(train_ngram_is, alpha, n)
        train_perplexity, val_perplexity = get_perplexity(probs, train_ngram_is, val_ngram_is)
        if report:
            print('alpha: {}, val_perplexity: {}, train_perplexity: {}'.format(
            alpha, val_perplexity, train_perplexity))

        if opt_perp > val_perplexity:
            opt_perp = val_perplexity
            opt_alpha = alpha
            opt_probs = probs

    return opt_probs, opt_alpha


def gen_interp_lambdas(step, n):
    lambda_perms = data_processing.perms(np.arange(0, 1, 1/step), n)
    return [lambdas for lambdas in lambda_perms if sum(lambdas) == 1]


def train_interp(train, val1, val2, alpha_range, n):

    # load 'configs' for each ngram setting
    train_ngram_configs = [data_processing.doc_to_ngram_indices(train, i+1, char_to_index) for i in range(n)]
    val1_ngram_configs = [data_processing.doc_to_ngram_indices(val1, i+1, char_to_index) for i in range(n)]

    # get optimal alphas
    opt_alphas, add_alpha_probs = [], []
    for i, train_ngram_is, val1_ngram_is in zip(range(n), train_ngram_configs, val1_ngram_configs):
        probs, alpha = train_add_alpha(train_ngram_is, val1_ngram_is, alpha_range, i+1, report=False)
        opt_alphas.append(alpha)
        add_alpha_probs.append(probs)
    
    

    lambda_configs = gen_interp_lambdas(10, n)

    val2_ngram_is = data_processing.doc_to_ngram_indices(val2, n, char_to_index)
    opt_perp, opt_lambdas = float('inf'), []
    for lambdas in lambda_configs:
        
        curr_probs = [p for p in add_alpha_probs]
        res_probs = np.zeros((num_chars,)*n)
        for j in range(n):
            res_probs += lambdas[j] * curr_probs[j]

        train_perp, test_perp = get_perplexity(res_probs, train_ngram_is, val2_ngram_is)

        print('lambdas: {}, test_perp: {}, train_perp: {}'.format(lambdas, test_perp, train_perp))
 
        if opt_perp > test_perp:
            opt_perp = test_perp
            opt_lambdas = lambdas
            opt_probs = res_probs

    # opt_perp, opt_alpha = float('inf'), 0
    # opt_lambdas, opt_probs = [], []
    # for alpha in alpha_range:
    #     for lambdas in lambda_configs:
    #         probs = np.zeros((num_chars,)*n)
    #         for i, lambda_i in zip(range(n), lambdas):
    #             train_is, val_is = train_ngram_configs[i], val_ngram_configs[i]
    #             prob_i = add_alpha_vec(train_is, alpha, i+1)
    #             probs += lambda_i * prob_i

    #             train_perplexity, val_perplexity = test_perplexity(probs, train_is, val_is)

    #         print('Alpha: {}, lambdas: {}, val_perp: {}, train_perp: {}'.format(alpha, lambdas, val_perplexity, train_perplexity))
        
    #         if opt_perp > val_perplexity:
    #             opt_perp = val_perplexity
    #             opt_lambdas = lambdas
    #             opt_probs = probs
    #             opt_alpha = alpha_range

    return opt_probs, opt_lambdas, opt_alphas
