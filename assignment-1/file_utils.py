from collections import defaultdict
import re

charset_rgx = r'[^a-zA-Z\d .]'
digits_rgx = r'\d'


def save_model(probs, lang):
    f = open('data/model.'+lang, 'w+')
    for k, v in sorted(probs.items()):
        for c, p in sorted(v.items()):
            f.write('{}${}\n'.format(k+c, p))
    f.close()


def read_model(lang):
    f = open('data/model.'+lang, 'r')
    probs = defaultdict(dict)
    for line in f:
        k, v = line.split('$')
        c12, c3 = k[:2], k[2]
        if c12 in probs:
            probs[c12][c3] = float(v)
        else:
            new_d = {}
            new_d[c3] = float(v)
            probs[c12] = new_d
    return probs


def map_to_index(ngram, indices):
    c0, c1, c2 = ngram
    return (indices[c0], indices[c1], indices[c2])


def map_to_char(index):
    print('not implemented')


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
