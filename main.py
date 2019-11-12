import sys
import argparse
import numpy as np

import data_processing, file_utils, lang_model
from argument_parser import get_args
from training_helper import add_alpha_training_vec, interp_training_vec

training_routines_dict = {'add_alpha': add_alpha_training_vec, 
                            'interpolation': interp_training_vec}
              
args = get_args()
task = args.task

if task == 'train':

    infile = args.training_file
    lang = args.language
    train_type = args.train_type
    n = args.n
    format = args.format

    docs = file_utils.read_file(infile, n)
    N = len(docs)

    np.random.shuffle(docs)

    # split data
    tr_i, val_i = int(N*.8), int(N*.9)
    train, val1, val2 = docs[:tr_i], docs[tr_i:val_i], docs[val_i:]

    training_routine = training_routines_dict[train_type]
    probs = training_routine(train, val1, val2, n)

    # save model in specified format
    file_utils.save_model(format, probs, lang, n)
    print('\'{}\' model saved to data folder.'.format(format))

elif task == 'generate':

    lang = args.language
    n = args.n
    format = args.format

    probs = file_utils.read_model(format, lang, n)
    w_gen = lang_model.generate_from_LM(300, probs, n)
    print(w_gen)

elif task == 'perp':

    infile = args.document_file
    lang = args.language
    n = args.n
    format = args.format

    # read in trained model and calculate perplexity
    probs = file_utils.read_model(format, lang, n)
    doc = file_utils.read_file(infile, n)
    perplexity = lang_model.get_perplexity_from_doc(doc, n, probs)
    print('Perplexity: {}'.format(perplexity))
