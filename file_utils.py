from collections import defaultdict
import re
import numpy as np

import data_processing
from const import *

def save_model_display(probs, lang, n, charset, indices, separator):
    ngrams = data_processing.perms(charset, n)
    ngram_is = data_processing.ngrams_to_indices(ngrams, indices)

    with open('data/model-display.{}.{}'.format(lang, n), 'w+') as f:
        for ngram, indices in zip(ngrams, ngram_is):
            p = probs[tuple(indices.T)]
            ngram_s = ''.join(ngram)
            f.write('{}{}{}\n'.format(ngram_s, separator, p))


def read_model_display(lang, shape, n, char_to_index, separator, name_stem='data/model-display.{}.{}'):
    p_mat = np.zeros(shape)
    with open(name_stem.format(lang, n), 'r') as f:
        for line in f:
            k, v = line.split(separator)
            k = [char_to_index[c] for c in k]
            p_mat[tuple(np.transpose(k))] = float(v)

    # handle zero entries so that probs sum to 1 always
    p_mat[p_mat == 0] = 1e-10
    s = np.sum(p_mat, axis=n-1)
    if np.ndim(s) == 0:
        s = 1 if s == 0 else s
    else :
        s[s == 0] = 1
    den = np.stack([s for _ in range(len(char_to_index))], axis=n-1)
    probs = p_mat/den
    return probs


def save_model_vec(probs, lang, n):
    np.savez('data/model-vec.{}.{}'.format(lang, n), probs=probs)


def read_model_vec(lang, shape, n):
    p = np.load('data/model-vec.{}.{}.npz'.format(lang, n))
    probs = p['probs']
    return probs.reshape(shape)


def save_model(format, probs, lang, n):
    if format == 'numpy':
        save_model_vec(probs, lang, n)
    elif format == 'normal':
        save_model_display(probs, lang, n, charset, char_to_index, separator)


def read_model(format, lang, n):
    shape = (num_chars,) * n
    if format == 'numpy':
        probs = read_model_vec(lang, shape, n)
    elif format == 'normal':
        probs = read_model_display(
            lang, shape, n, char_to_index, separator)

    return probs


def preprocess_line(line, n):
    filtered_chars = re.sub(charset_rgx, '', line)
    replaced_nums = re.sub(digits_rgx, '0', filtered_chars)
    lc = replaced_nums.lower()
    lc = '#'*(n-1) + lc + '#'
    return lc


def read_file(infile, n):
    docs = []
    with open(infile) as f:
        for line in f:
            line = preprocess_line(line, n)
            docs.append(line)
    return docs
