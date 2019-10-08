import sys
import file_utils
import lang_model
import data_processing
import numpy as np

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
    alpha_range = [1/(1.2**i) for i in range(20)]

    # split data
    tr_i, val_i = int(N*.8), int(N*.9)
    train, val, test = docs[:tr_i], docs[tr_i:val_i], docs[val_i:]

    # get optimum model through alpha grid search, perform test and save
    probs, alpha = lang_model.train_model(train, val, alpha_range)
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
    w_gen = lang_model.generate_from_LM(300, probs)
    print(w_gen)

elif task == 'perp':

    if argnum != 2:
        print('Calculating document perplexity needs 2 arguments, got {}'.format(argnum))
        sys.exit()

    infile = sys.argv[2]
    lang = sys.argv[3]

    f_doc = data_processing.to_string(file_utils.read_file(infile))
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
