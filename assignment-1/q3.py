import re
import sys
import re
from random import random, choice
from math import log
from collections import defaultdict
import numpy as np
import utils
import itertools

num_chars = 29
d = num_chars ** 3
charset = ' .0abcdefghijklmnopqrstuvwxyz'
indices = {c:i for i, c in enumerate(charset)}

# generate csv of indices


def save_model(probs, lang):
    f = open('model.'+lang, 'w+')
    for k, v in probs:
        f.write('{} {}'.format(k, v))
    f.close()


def read_model(lang):
    f = open('data/model-br.'+lang, 'r')
    probs = defaultdict(float)
    for line in f:
        k, v = line.split('\t')
        probs[k] = float(v)

    return probs


def map_to_index(ngram):
    c0, c1, c2 = ngram
    return (indices[c0], indices[c1], indices[c2])


def map_to_char(index):
    print('not implemented')


def generate_from_LM(N, probs):
    c1 = charset[np.random.randint(0, num_chars-1)]    
    c2 = charset[np.random.randint(0, num_chars-1)]

    gen_lst = [0 for _ in range(N+2)]
    gen_lst[0] = c1
    gen_lst[1] = c2
    for i in range(N):
        bigram = str(c1) + str(c2)
        probs_bigram = probs[bigram]
        sorted_tuples = sorted(probs_bigram.items(), key=lambda x:x[1])
        max_p = sorted_tuples[-1]

        j = len(sorted_tuples)
        for k in reversed(range(len(sorted_tuples))):
            if sorted_tuples[k] == max_p:
                j-=1
            else :
                break

        rdint = np.random.randint(j, len(sorted_tuples))    
        max_tuple = sorted_tuples[rdint]

        max_char = max_tuple[0]
        gen_lst[i+2] = (str(max_char))

        c1 = c2
        c2 = max_char

    return ''.join(gen_lst)


def add_alpha(docs, alpha):    
    trigram_counts = defaultdict(float)
    trigram_smoothed = defaultdict(dict)
    bigram_counts = defaultdict(float)
    for doc in docs:
        ngrams = utils.get_ngrams(doc, 3)
        bigrams = utils.get_ngrams(doc, 2)

        for ngram in ngrams:
            trigram_counts[ngram] += 1

        for bigram in bigrams:
            bigram_counts[bigram] += 1

    
    # add smoothing
    # trigrams
    alpha_d = alpha * d
    for c1 in charset:
        for c2 in charset:
            for c3 in charset:
                bigram = c1+c2
                trigram = bigram + c3


                addend = (trigram_counts[trigram] + alpha) / (bigram_counts[bigram] + alpha_d)
                nested_d = trigram_smoothed[bigram]
                if c3 in nested_d:
                    nested_d[c3] += addend
                else :
                    nested_d[c3] = addend 

                trigram_smoothed[bigram] = nested_d

    return trigram_smoothed


def train_model(train, val, alpha_range):
    max_alpha = alpha_range[0]
    curr_prob = 0

    # get ngrams for each document of validation set and flatten array
    val_ngrams_2d = [utils.get_ngrams(val_sen, 3) for val_sen in val]
    val_ngrams = list(itertools.chain(*val_ngrams_2d))

    probs = []

    for alpha in alpha_range:
        probs = add_alpha(train, alpha)

        res = probs.values()
        # sum = 0
        # for c in res : sum += c
        # print(sum)

        # val_probs = [probs[i] for i in val_ngrams]
        # sum_probs = np.sum(val_probs)
        # if curr_prob < sum_probs:
        #     curr_prob, max_alpha = sum_probs, alpha

    return probs
    
##########################

tri_counts = defaultdict(int)  # counts of all trigrams in input
charset_rgx = r'[^a-zA-Z\d .]'
digits_rgx = '[0-9]'


def preprocess_line(line):
    filtered_chars = re.sub(charset_rgx, '', line)
    replaced_nums = re.sub(digits_rgx, '0', filtered_chars)
    lc = replaced_nums.lower()
    return lc


# here we make sure the user provides a training filename when
# calling this program, otherwise exit with a usage error.
if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

infile = sys.argv[1]  # get input argument: the training file
lang = infile.split('.')[-1]

# This bit of code gives an example of how you might extract trigram counts
# from a file, line by line. If you plan to use or modify this code,
# please ensure you understand what it is actually doing, especially at the
# beginning and end of each line. Depending on how you write the rest of
# your program, you may need to modify this code.
docs = []
split = 0.8, .1, .1
with open(infile) as f:
    prev_chars = ""
    for line in f:
        # keep context from prev line
        line = prev_chars + line
        line = preprocess_line(line)
        docs.append(line)
        for j in range(len(line)-(3)):
            trigram = line[j:j+3]
            tri_counts[trigram] += 1
        prev_chars = line[-2:]

# np.random.seed(10)
np.random.shuffle(docs)
N = len(docs)
# alpha_range = [.1, .01, .001 ,.0001, .00001]
alpha_range = [.01]

tr_i, val_i, te_i = int(N*.8), int(N*.9), int(N)
train_s, val, test = docs[:tr_i], docs[tr_i:val_i], docs[val_i:]

probs = train_model(train_s, val, alpha_range)
print(generate_from_LM(300, probs))

# Some example code that prints out the counts. For small input files
# the counts are easy to look at but for larger files you can redirect
# to an output file (see Lab 1).
# print("Trigram counts in ", infile, ", sorted alphabetically:")
# for trigram in sorted(tri_counts.keys()):
#     print(trigram, ": ", tri_counts[trigram])
# print("Trigram counts in ", infile, ", sorted numerically:")
# for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
#     print(tri_count[0], ": ", str(tri_count[1]))
