charset_rgx = r'[^a-zA-Z\d .]'
digits_rgx = r'\d'
separator = '\t'
charset = ' .0abcdefghijklmnopqrstuvwxyz#'
num_chars = len(charset)
char_to_index = {c: i for i, c in enumerate(charset)}
