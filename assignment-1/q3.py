import re
import sys
import re
from random import random
from math import log
from collections import defaultdict
import numpy as np
import utils
import itertools

num_chars = 29
d = num_chars ** 3

indices = {}
indices[' '] = 0
indices['.'] = 1
indices['0'] = 2
for i in range(ord('a'), ord('z')+1):
    indices[chr(i)] = i - ord('a') + 3 

# generate csv of indices
def gen_charset_csv(indices: dict):
    print('not implemented!')

def save_model(probs, lang):
    np.savez('model-q3-{}'.format(lang), probs)

def map_to_index(ngram):
    c0, c1, c2 = ngram
    # print(ngram)
    return (indices[c0], indices[c1], indices[c2])

def map_to_char(index):
    print('not implemented')

def add_alpha(docs, alpha):
    probs = defaultdict(float)
    denoms = defaultdict(float)
    for doc in docs:
        ngrams = utils.get_ngrams(doc, 3)
        for ngram in ngrams: 
            probs[ngram] += 1
            
            # third (and 'most recent') char is summed out
            denoms[ngram[:-1]] += 1
    
    # add smoothing
    for ngram, _ in probs.items():
        probs[ngram] += alpha
    
    alpha_d = alpha * d
    for c1c2, _ in denoms.items():
        denoms[c1c2] += alpha_d

    # estimate probs
    for ngram, num in probs.items():
        probs[ngram] = num/denoms[ngram[:-1]]

    return probs

    

def add_alpha_vec(docs, alpha):
    # for each doc, get n grams
        # compute ngram counts + total count
    # add alpha to each
    # normalise
    N = 0
    probs = np.zeros((num_chars, num_chars, num_chars))
    for doc in docs:
        n_grams = utils.get_ngrams(doc, 3)
        for ngram in n_grams:
            i0, i1, i2 = map_to_index(ngram)
            probs[i0, i1, i2] += 1

    N = np.sum(probs, axis=2)
    # print(N)
    probs = probs + alpha
    den = N + alpha*(num_chars ** 3)
    den = np.stack([den for _ in range(num_chars)], axis=2)
    probs = probs / den

    return probs

def train_model(train, val, alpha_range):
    max_alpha = alpha_range[0]
    curr_prob = 0

    # get ngrams for each document of validation set and flatten array
    val_ngrams_2d = [utils.get_ngrams(val_sen, 3) for val_sen in val]
    val_ngrams = list(itertools.chain(*val_ngrams_2d))
    
    # convert to indices
    val_i = [map_to_index(ngram) for ngram in val_ngrams if len(ngram) == 3]
    for alpha in alpha_range:
        probs = add_alpha_vec(train, alpha)
        val_probs = [probs[i] for i in val_i]
        sum_probs = np.sum(val_probs)
        print('alpha = {}, sum_probs = {}'.format(alpha, sum_probs))
        if curr_prob < sum_probs:
            curr_prob, max_alpha = sum_probs, alpha

    print('max_alpha = {}'.format(max_alpha))

##########################

tri_counts=defaultdict(int) #counts of all trigrams in input
charset_rgx = r'[^a-zA-Z\d .]'
digits_rgx = '[0-9]'
def preprocess_line(line):
    filtered_chars = re.sub(charset_rgx, '', line) 
    replaced_nums = re.sub(digits_rgx, '0', filtered_chars)
    lc = replaced_nums.lower()
    return lc

#here we make sure the user provides a training filename when
#calling this program, otherwise exit with a usage error.
if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

infile = sys.argv[1] #get input argument: the training file
lang = infile.split('.')[-1]

#This bit of code gives an example of how you might extract trigram counts
#from a file, line by line. If you plan to use or modify this code,
#please ensure you understand what it is actually doing, especially at the
#beginning and end of each line. Depending on how you write the rest of
#your program, you may need to modify this code.
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

np.random.seed(10)
np.random.shuffle(docs)
N = len(docs)
alpha_range = np.arange(0.05, 2.05, 0.05)

tr_i, val_i, te_i = int(N*.8), int(N*.9), int(N)
train_s, val, test = docs[:tr_i], docs[tr_i:val_i], docs[val_i:]


train_model(train_s, val, alpha_range)
#Some example code that prints out the counts. For small input files
#the counts are easy to look at but for larger files you can redirect
#to an output file (see Lab 1).
# print("Trigram counts in ", infile, ", sorted alphabetically:")
# for trigram in sorted(tri_counts.keys()):
#     print(trigram, ": ", tri_counts[trigram])
# print("Trigram counts in ", infile, ", sorted numerically:")
# for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
#     print(tri_count[0], ": ", str(tri_count[1]))


