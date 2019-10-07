import re
import sys
import re
from random import random, choice, randint
from math import log
from collections import defaultdict
import numpy as np
import data_processing
import file_utils


charset = ' .0abcdefghijklmnopqrstuvwxyz#'
num_chars = len(charset)
d = num_chars ** 3
indices = {c: i for i, c in enumerate(charset)}

perms = data_processing.perms(list(charset), 3)


def generate_from_LM(N, probs):
    # c1 = charset[randint(0, num_chars - 1)]
    # c2 = charset[randint(0, num_chars - 1)]

    c1, c2 = '#', '#'
    gen_lst = [c1, c2]
    for _ in range(N):
        bigram = str(c1) + str(c2)
        # convert dict of 'continuations' to list of (char, prob) tuples
        probs_bigram = probs[bigram].items()

        # sort ascendingly by probability
        sorted_tuples = sorted(probs_bigram, key=lambda x: x[1])
        max_p = sorted_tuples[-1]

        j = len(sorted_tuples)
        for k in reversed(range(len(sorted_tuples))):
            if sorted_tuples[k] == max_p:
                j -= 1
            else:
                break

        if j == len(sorted_tuples) - 1:
            j -= np.random.randint(0, 5)

        rdint = randint(j, len(sorted_tuples) - 1)
        max_char = sorted_tuples[rdint][0]

        if max_char == '#':
            max_char = '\n'
        gen_lst.append(str(max_char))

        c1 = c2
        c2 = max_char

    return ''.join(gen_lst)[2:]


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

        addend = (trigram_counts[trigram] + alpha) / \
            (bigram_counts[bigram] + alpha_d)
        nested_d = trigram_smoothed[bigram]
        if c3 in nested_d:
            nested_d[c3] += addend
        else:
            nested_d[c3] = addend

        trigram_smoothed[bigram] = nested_d

    return trigram_smoothed


def train_model(train, val, alpha_range):
    # initialise to 'infinity'
    opt_perp, opt_alpha = float('inf'), float('inf')
    opt_probs = []
    for alpha in alpha_range:
        probs = add_alpha(train, alpha)

        val_f = data_processing.flatten(val)
        val_ngrams = data_processing.get_ngrams(val_f, 3)

        train_f = data_processing.flatten(train)
        train_ngrams = data_processing.get_ngrams(train_f, 3)

        val_perplexity = data_processing.perplexity(val_ngrams, probs)
        train_perplexity = data_processing.perplexity(train_ngrams, probs)
        print('alpha: {}, val_perplexity: {}, train_perplexity: {}'.format(
            alpha, val_perplexity, train_perplexity))

        if opt_perp > val_perplexity:
            opt_perp = val_perplexity
            opt_alpha = alpha
            opt_probs = probs

    return opt_probs, opt_alpha


if len(sys.argv) <= 2:
    print('Usage: ', sys.argv[0])
    print('        train    <training_file> <language>')
    print('        generate <language>')
    print('        perp     <document_file> <language>')
    sys.exit()


task = sys.argv[1]  # 'train' or 'generate'
# 'remove' script name and task type 'e.g. q5.py train'
argnum = len(sys.argv) - 2

if task == 'train':

    if argnum != 2:
        print('Training needs 2 arguments, got {}'.format(argnum))
        sys.exit()

    infile = sys.argv[2]
    lang = sys.argv[3]

    docs = file_utils.read_file(infile)
    N = len(docs)

    # np.random.seed(10)
    np.random.shuffle(docs)
    # alpha_range=[.00001, .0001, .001, .01, .1]
    # alpha_range = [i/20 for i in range(1, 21)]
    alpha_range = [1/(1.2**i) for i in range(20)]

    # split data
    tr_i, val_i = int(N*.8), int(N*.9)
    train, val, test = docs[:tr_i], docs[tr_i:val_i], docs[val_i:]

    # get optimum model through alpha grid search, perform test and save
    probs, alpha = train_model(train, val, alpha_range)
    test_perplexity = data_processing.perplexity(test, probs)
    print('******** RESULT ********')
    print('Alpha:           {}'.format(alpha))
    print('Test perplexity: {}'.format(test_perplexity))
    print('************************')

    file_utils.save_model(probs, lang)
    print('Model saved at \'data/model.{}\''.format(lang))

elif task == 'generate':

    if argnum != 1:
        print('Generating needs 1 argument, got {}'.format(argnum))

    lang = sys.argv[2]
    probs = file_utils.read_model(lang)
    w_gen = generate_from_LM(300, probs)
    print(w_gen)

elif task == 'perp':

    if argnum != 2:
        print('Calculating document perplexity needs 2 arguments, got {}'.format(argnum))

    infile = sys.argv[2]
    lang = sys.argv[3]

    f_doc = data_processing.flatten(file_utils.read_file(infile))
    doc_ngrams = data_processing.get_ngrams(f_doc, 3)
    probs = file_utils.read_model(lang)

    perplexity = data_processing.perplexity(doc_ngrams, probs)
    print('Perplexity: {}'.format(perplexity))

else:
    print('Task must be \'train\', \'generate\' or \'perplexity\'')

# grid test for alpha in each language model against 10% validation set
# compute perplexity (sum of -log2 ) for the test document for each optimum model
# - right after training
#
