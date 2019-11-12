import data_processing, lang_model, file_utils
from const import char_to_index

def add_alpha_training_vec(train, val, test, n):
    alpha_range = [(2**i)/2**20 for i in range(21)]

    # get optimal model through alpha grid search, perform test and save
    train_ngram_idxs = data_processing.doc_to_ngram_indices(
        train, n, char_to_index)
    val_ngram_idxs = data_processing.doc_to_ngram_indices(
        val, n, char_to_index)
    probs, alpha = lang_model.train_add_alpha(
        train_ngram_idxs, val_ngram_idxs, alpha_range, n)

    test_ngram_idxs = data_processing.doc_to_ngram_indices(
        test, n, char_to_index)
    test_perplexity = lang_model.perplexity(test_ngram_idxs, probs)

    print('******** RESULT ********')
    print(f'Alpha:           {alpha}')
    print(f'Test perplexity: {test_perplexity}')
    print('************************')

    return probs

def interp_training_vec(train, val1, val2, n):
    alpha_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    # arrays of arrays of ngram indices up to n
    train_ngram_configs = [data_processing.doc_to_ngram_indices(
        train, i+1, char_to_index) for i in range(n)]
    val1_ngram_configs = [data_processing.doc_to_ngram_indices(
        val1, i+1, char_to_index) for i in range(n)]

    # this is an array of ngram indices
    val2_ngram_idxs = data_processing.doc_to_ngram_indices(
        val2, n, char_to_index)

    probs, lambdas, alpha = lang_model.train_interp(
        train_ngram_configs, val1_ngram_configs, val2_ngram_idxs, alpha_range, n)

    test = file_utils.read_file('data/test', n)
    test_ngram_idxs = data_processing.doc_to_ngram_indices(
        test, n, char_to_index)

    test_perp = lang_model.perplexity(test_ngram_idxs, probs)
    print('******** RESULT ********')
    print(f'Lambdas:          {lambdas}')
    print(f'Alphas:           {alpha}')
    print(f'Test perplexity:  {test_perp}')
    print('************************')

    return probs

