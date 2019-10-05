import re
import sys
import re
from random import random, choice, randint
from math import log
from collections import defaultdict
import numpy as np
import data_processing, file_utils
import itertools

num_chars = 29
d = num_chars ** 3

charset = ' .0abcdefghijklmnopqrstuvwxyz'
indices = {c:i for i, c in enumerate(charset)}

perms = list(itertools.product(*([charset]*3)))

def generate_from_LM(N, probs):
    c1 = charset[randint(0, num_chars - 1)]
    c2 = charset[randint(0, num_chars - 1)]

    gen_lst = [c1, c2]
    for _ in range(N):
        bigram = str(c1) + str(c2)
        # convert dict of 'continuations' to list of (char, prob) tuples
        probs_bigram = probs[bigram].items()

        # sort ascendingly by probability
        sorted_tuples = sorted(probs_bigram, key=lambda x:x[1])
        max_p = sorted_tuples[-1]

        j = len(sorted_tuples)
        for k in reversed(range(len(sorted_tuples))):
            if sorted_tuples[k] == max_p:
                j-=1
            else :
                break
        
        if j == len(sorted_tuples) - 1:
            j -= np.random.randint(0, 5)    

        rdint = randint(j, len(sorted_tuples) - 1)    
        max_char = sorted_tuples[rdint][0]

        gen_lst.append(str(max_char))

        c1 = c2
        c2 = max_char

    return ''.join(gen_lst)


def add_alpha(docs, alpha):    
    trigram_counts = defaultdict(float)
    trigram_smoothed = defaultdict(dict)
    bigram_counts = defaultdict(float)
    for doc in docs:
        ngrams = data_processing.get_ngrams(doc, 3)
        bigrams = data_processing.get_ngrams(doc, 2)

        for ngram in ngrams:
            trigram_counts[ngram] += 1

        for bigram in bigrams:
            bigram_counts[bigram] += 1

    
    # add smoothing
    # trigrams
    alpha_d = alpha * num_chars
    for c1, c2, c3 in perms:
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
    val_ngrams_2d = [data_processing.get_ngrams(val_sen, 3) for val_sen in val]
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
    

if len(sys.argv) <= 2 :
    print('Not enough arguments!')
    sys.exit()


task = sys.argv[1] # 'train' or 'generate'
 # remove script name and task type 'e.g. q5.py train'
argnum = len(sys.argv) - 2

if task == 'train':

    if argnum != 2:
        print('Training needs 2 arguments, got {}'.format(argnum))
        sys.exit()

    infile = sys.argv[2] 
    lang = sys.argv[3]

    docs = file_utils.read_file(infile, 300)
    N = len(docs)
    # np.random.seed(10)
    np.random.shuffle(docs)

    alpha_range=[.00001, .0001, .001, .01, .1]


    # split data
    tr_i, val_i = int(N*.8), int(N*.9)
    train, val, test = docs[:tr_i], docs[tr_i:val_i], docs[val_i:]

    probs = train_model(train, val, alpha_range)
    file_utils.save_model(probs, lang)

    print('Model saved at \'data/model.{}\''.format(lang))

elif task == 'generate':

    if argnum != 1:
        print('Generating needs 1 argument, got {}'.format(argnum))

    lang = sys.argv[2]
    probs = file_utils.read_model(lang)
    w_gen = generate_from_LM(300, probs)

    print(w_gen)
    print(data_processing.perplexity(w_gen, 3, probs))

else : print('Task must either be \'train\' or \'generate\'')