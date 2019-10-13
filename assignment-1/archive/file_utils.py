from collections import defaultdict
import re
import data_processing
import numpy as np

charset_rgx = r'[^a-zA-Z\d .]'
digits_rgx = r'\d'
separator = '\t'

def save_model_display(probs, lang, n, charset, indices):
    ngrams = data_processing.perms(charset, n)    
    ngram_is = data_processing.ngrams_to_indices(ngrams, indices)

    with open('data/model-display.{}.{}'.format(lang, n), 'w+') as f:
        for ngram, indices in zip(ngrams, ngram_is):
            p = probs[tuple(indices.T)]
            ngram_s = ''.join(ngram)
            f.write('{}{}{}\n'.format(ngram_s, separator, p))


def save_model_vec(probs, lang, n):
    np.savez('data/model-vec.{}.{}'.format(lang, n), probs=probs)


def read_model_vec(lang, shape, n):
    p = np.load('data/model-vec.{}.{}.npz'.format(lang, n))
    probs = p['probs']
    return probs.reshape(shape)


def read_model_vec_reformat(lang, shape, char_to_index, name_stem='data/model-br.'):
    p_mat = np.zeros(shape)
    with open(name_stem+lang, 'r') as f:
        for line in f:
            k, v = line.split(separator)
            k = [char_to_index[c] for c in k]
            p_mat[tuple(np.transpose(k))] = float(v)

    p_mat[p_mat == 0] = 1e-10
    s = np.sum(p_mat, axis=2)
    s[s == 0] = 1
    den = np.stack([s for _ in range(30)], axis=2)
    probs = p_mat/den
    return probs


def save_model(probs, lang, name_stem='data/model.'):
    with open(name_stem+lang, 'w+') as f:
        for k, v in sorted(probs.items()):
            for c, p in sorted(v.items()):
                f.write('{}{}{}\n'.format(k+c, separator, p))


def read_model(lang, name_stem='data/model-br.'):
    probs = defaultdict(dict)
    with open(name_stem+lang, 'r') as f:
        for line in f:
            k, v = line.split(separator)
            c12, c3 = k[:2], k[2]
            if c12 in probs:
                probs[c12][c3] = float(v)
            else:
                new_d = {}
                new_d[c3] = float(v)
                probs[c12] = new_d
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
