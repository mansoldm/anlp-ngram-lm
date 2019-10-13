import sys
import file_utils_final
import lang_model_final
from lang_model_final import charset, char_to_index, num_chars
import data_processing_final
import numpy as np


def add_alpha_training_vec(train, val, test, n):
    alpha_range = [(2**i)/2**20 for i in range(21)]

    # get optimum model through alpha grid search, perform test and save
    train_ngram_is = data_processing_final.doc_to_ngram_indices(train, n, char_to_index)
    val_ngram_is = data_processing_final.doc_to_ngram_indices(val, n, char_to_index)
    probs, alpha = lang_model_final.train_add_alpha(train_ngram_is, val_ngram_is, alpha_range, n)

    test_ngram_is = data_processing_final.doc_to_ngram_indices(test, n, char_to_index)
    test_perplexity = data_processing_final.perplexity(test_ngram_is, probs)

    print('******** RESULT ********')
    print('Alpha:           {}'.format(alpha))
    print('Test perplexity: {}'.format(test_perplexity))
    print('************************')

    return probs


def interp_training_vec(train, val1, val2, n):
    alpha_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    # these are arrays of arrays of ngram indices up to 'n' (in the method signature)
    train_ngram_configs = [data_processing_final.doc_to_ngram_indices(
        train, i+1, char_to_index) for i in range(n)]
    val1_ngram_configs = [data_processing_final.doc_to_ngram_indices(
        val1, i+1, char_to_index) for i in range(n)]
    
    # this is an array of ngram indices
    val2_ngram_is = data_processing_final.doc_to_ngram_indices(
        val2, n, char_to_index)

    probs, lambdas, alpha = lang_model_final.train_interp(train_ngram_configs, val1_ngram_configs, val2_ngram_is, alpha_range, n)

    test = file_utils_final.read_file('data/test', n)
    test_ngram_is = data_processing_final.doc_to_ngram_indices(test, n, char_to_index)

    test_perp = data_processing_final.perplexity(test_ngram_is, probs)
    print('******** RESULT ********')
    print('Lambdas:          {}'.format(lambdas))
    print('Alphas:            {}'.format(alpha))
    print('Test perplexity:  {}'.format(test_perp))
    print('************************')

    return probs

    
train_dict = {'add_alpha': add_alpha_training_vec, 'interpolation': interp_training_vec}

if len(sys.argv) <= 2:
    print('Usage: ', sys.argv[0])
    print('        train    <training_file> <language> <train_type> <n> <format>')
    print('        generate <language> <n> <format>')
    print('        perp     <document_file> <language> <n>')
    sys.exit()


task = sys.argv[1]  # 'train' or 'generate'
# 'remove' script name and task type 'e.g. q5.py train'
argnum = len(sys.argv) - 2

if task == 'train':

    if argnum != 5:
        print('Training needs 4 arguments, got {}'.format(argnum))
        sys.exit()

    infile = sys.argv[2]
    lang = sys.argv[3]
    train_type = sys.argv[4]
    n = int(sys.argv[5])
    format = sys.argv[6]

    if train_type not in train_dict:
        print('Training type must be either \'add_alpha\' or \'interpolation\'')
        sys.exit()
    if n < 1 : 
        print('The value of n must be at least 1, got {}'.format(n))
        sys.exit()

    docs = file_utils_final.read_file(infile, n)
    N = len(docs)

    # np.random.seed(10)
    np.random.shuffle(docs)

    # split data
    tr_i, val_i = int(N*.8), int(N*.9)
    train, val1, val2 = docs[:tr_i], docs[tr_i:val_i], docs[val_i:]

    train_f = train_dict[train_type]

    probs = train_f(train, val1, val2, n)
    if format == 'numpy':
        file_utils_final.save_model(probs, lang, n)
    elif format == 'normal':
        file_utils_final.save_model_display(probs, lang, n, charset, char_to_index)
    else :
        print('Format should be \'numpy\' or \'normal\', got {}'.format(format))
        sys.exit()
 
    print('Model saved in data folder.'.format(lang, n))

elif task == 'generate':

    if argnum != 3:
        print('Generating needs 3 arguments, got {}'.format(argnum))
        sys.exit()

    lang = sys.argv[2]
    n = int(sys.argv[3])
    format = sys.argv[4]

    if n < 1 : 
        print('The value of n must be at least 1, got {}'.format(n))
        sys.exit()

    shape = (num_chars,) * n

    probs = []
    if format == 'numpy':
        probs = file_utils_final.read_model(lang, shape, n)
    elif format == 'normal':
        probs = file_utils_final.read_model_display(lang, shape, n, char_to_index)
    else :
        print('Format should be \'numpy\' or \'normal\', got {}'.format(format))
        sys.exit()
    w_gen = lang_model_final.generate_from_LM(300, probs, n, char_to_index, num_chars, charset)
    print(w_gen)

elif task == 'perp':

    if argnum != 4:
        print('Calculating document perplexity needs 4 arguments, got {}'.format(argnum))
        sys.exit()

    infile = sys.argv[2]
    lang = sys.argv[3]
    n = int(sys.argv[4])
    format = sys.argv[5]

    if n < 1 : 
        print('The value of n must be at least 1, got {}'.format(n))
        sys.exit()
    
    shape = (num_chars,) * n

    doc = file_utils_final.read_file(infile, n) 
    doc_ngram_is = data_processing_final.doc_to_ngram_indices(doc, n, char_to_index)
    probs = []
    if format == 'numpy':
        probs = file_utils_final.read_model(lang, shape, n)
    elif format == 'normal':
        probs = file_utils_final.read_model_display(lang, shape, n, char_to_index)
    else :
        print('Format should be \'numpy\' or \'normal\', got {}'.format(format))
        sys.exit()

    probs = file_utils_final.read_model(lang, shape, n)

    perplexity = data_processing_final.perplexity(doc_ngram_is, probs)
    print('Perplexity: {}'.format(perplexity))

else:
    print('Task must be \'train\', \'generate\' or \'perp\'')

# grid test for alpha in each language model against 10% validation set
# compute perplexity (sum of -log2 ) for the test document for each optimum model
# - right after training
#
