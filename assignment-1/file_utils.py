from collections import defaultdict
import re
import data_processing
import numpy as np

charset_rgx = r'[^a-zA-Z\d .]'
digits_rgx = r'\d'
separator = '\t'

def save_model_vec(probs, lang):
    with open('data/model-vec.{}'.format(lang), 'w+') as outfile:
        for prob in probs:
            np.savetxt(outfile, prob)
            outfile.write('\n')


def read_model_vec(lang, shape):
    probs = np.loadtxt('data/model-vec.{}'.format(lang))
    return probs.reshape(shape)


def save_model(probs, lang):
    with open('data/model.'+lang, 'w+') as f:
        for k, v in sorted(probs.items()):
            for c, p in sorted(v.items()):
                f.write('{}{}{}\n'.format(k+c, separator, p))


def read_model(lang):
    probs = defaultdict(dict)
    with open('data/model.'+lang, 'r') as f:
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


def preprocess_line(line):
    filtered_chars = re.sub(charset_rgx, '', line)
    replaced_nums = re.sub(digits_rgx, '0', filtered_chars)
    lc = replaced_nums.lower()
    lc = '##' + lc + '#'
    return lc


def read_file(infile):
    docs = []
    with open(infile) as f:
        for line in f:
            line = preprocess_line(line)
            docs.append(line)
    return docs
