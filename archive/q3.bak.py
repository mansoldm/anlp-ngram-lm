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
    den = N + alpha * (num_chars ** 3)
    den = np.stack([den for _ in range(num_chars)], axis=2)
    probs = probs / den

    return probs