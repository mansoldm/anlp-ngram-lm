import sys
import file_utils
import lang_model
import lang_model_vec
from lang_model_vec import charset, char_to_index, num_chars
import data_processing
import numpy as np


def add_alpha_training_vec(train, val, test, n):
    alpha_range = [(2**i)/2**20 for i in range(21)]

    # get optimum model through alpha grid search, perform test and save
    train_ngram_is = data_processing.doc_to_ngram_indices(train, n, char_to_index)
    val_ngram_is = data_processing.doc_to_ngram_indices(val, n, char_to_index)
    probs, alpha = lang_model_vec.train_add_alpha(train_ngram_is, val_ngram_is, alpha_range, n)

    test_ngram_is = data_processing.doc_to_ngram_indices(test, n, char_to_index)
    test_perplexity = data_processing.perplexity_vec(test_ngram_is, probs)

    print('******** RESULT ********')
    print('Alpha:           {}'.format(alpha))
    print('Test perplexity: {}'.format(test_perplexity))
    print('************************')

    return probs


def interp_training_vec(train, val, test, n):
    alpha_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    probs, lambdas, alpha = lang_model_vec.train_interp(train, val, test, alpha_range, n)

    test_f = data_processing.to_string(test)
    test_ngrams = data_processing.get_ngrams(test_f, n)
    test_ngram_is = data_processing.ngrams_to_indices(test_ngrams, char_to_index)

    test_perplexity = data_processing.perplexity_vec(test_ngram_is, probs)
    print('******** RESULT ********')
    print('Lambdas:          {}'.format(lambdas))
    print('Alphas:            {}'.format(alpha))
    print('Test perplexity:  {}'.format(test_perplexity))
    print('************************')

    return probs

    
train_dict = {'add_alpha': add_alpha_training_vec, 'interpolation': interp_training_vec}

if len(sys.argv) <= 2:
    print('Usage: ', sys.argv[0])
    print('        train    <training_file> <language> <train_type> <n>')
    print('        generate <language> <n>')
    print('        perp     <document_file> <language> <n>')
    sys.exit()


task = sys.argv[1]  # 'train' or 'generate'
# 'remove' script name and task type 'e.g. q5.py train'
argnum = len(sys.argv) - 2

if task == 'train':

    if argnum != 4:
        print('Training needs 4 arguments, got {}'.format(argnum))
        sys.exit()

    infile = sys.argv[2]
    lang = sys.argv[3]
    train_type = sys.argv[4]
    n = int(sys.argv[5])

    if train_type not in train_dict:
        print('Training type must be either \'add_alpha\' or \'interpolation\'')
        sys.exit()
    if n < 1 : 
        print('The value of n must be at least 1, got {}'.format(n))
        sys.exit()

    docs = file_utils.read_file(infile)
    N = len(docs)

    # np.random.seed(10)
    np.random.shuffle(docs)

    # split data
    tr_i, val_i = int(N*.8), int(N*.9)
    train, val, test = docs[:tr_i], docs[tr_i:val_i], docs[val_i:]

    train_f = train_dict[train_type]

    probs = train_f(train, val, test, n)

    file_utils.save_model_vec(probs, lang, n)
    print('Model saved at \'data/model-vec.{}{}\''.format(lang, n))

elif task == 'generate':

    if argnum != 2:
        print('Generating needs 2 arguments, got {}'.format(argnum))
        sys.exit()

    lang = sys.argv[2]
    n = int(sys.argv[3])
    if n < 1 : 
        print('The value of n must be at least 1, got {}'.format(n))
        sys.exit()

    shape = (num_chars,) * n

    probs = file_utils.read_model_vec(lang, shape, n)
    w_gen = lang_model_vec.generate_from_LM_vec(300, probs, n)
    print(w_gen)

elif task == 'perp':

    if argnum != 3:
        print('Calculating document perplexity needs 3 arguments, got {}'.format(argnum))
        sys.exit()

    infile = sys.argv[2]
    lang = sys.argv[3]
    n = int(sys.argv[4])
    if n < 1 : 
        print('The value of n must be at least 1, got {}'.format(n))
        sys.exit()
    
    shape = (num_chars,) * n

    f_doc = data_processing.to_string(file_utils.read_file(infile))
    doc_ngrams = data_processing.get_ngrams(f_doc, n)
    doc_ngram_is = data_processing.ngrams_to_indices(doc_ngrams, char_to_index)
    probs = file_utils.read_model_vec(lang, shape, n)

    perplexity = data_processing.perplexity_vec(doc_ngram_is, probs)
    print('Perplexity: {}'.format(perplexity))

else:
    print('Task must be \'train\', \'generate\' or \'perp\'')

# grid test for alpha in each language model against 10% validation set
# compute perplexity (sum of -log2 ) for the test document for each optimum model
# - right after training
#
