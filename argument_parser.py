import argparse


def check_ngram_length(n):
    try: 
        value = int(n)
    except ValueError:
        value = -1

    if value <= 0:
        return argparse.ArgumentTypeError(f'The N-gram length (--n) must be a strictly positive integer, got {n}')

    return value


def get_args():
    parser = argparse.ArgumentParser()

    # add subparsers for train, generate, perp tasks
    subparsers = parser.add_subparsers(dest='task')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--training_file', type=str,                required=True, help='Name of file/document to train the model with')
    train_parser.add_argument('--language',      type=str,                required=True, help='Language of training document')
    train_parser.add_argument('--train_type',    type=str,                help='Specify how to train the model (add_alpha, interpolation)', choices={'add_alpha', 'interpolation'}, default='add_alpha')
    train_parser.add_argument('--n',             type=check_ngram_length, required=True, help='N-gram length')
    train_parser.add_argument('--format',        type=str,                help='Whether to save model as human-readable or npz', choices={'normal', 'numpy'}, default='numpy')

    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('--language', type=str,                required=True, help='Language the model was trained on')
    generate_parser.add_argument('--n',        type=check_ngram_length, required=True, help='N-gram length')
    generate_parser.add_argument('--format',   type=str,                help='Format used to store the model', choices={'normal', 'numpy'}, default='numpy')

    perp_parser = subparsers.add_parser('perp')
    perp_parser.add_argument('--document_file', type=str,                required=True, help='Name of test file/document to calculate the perplexity of the model')
    perp_parser.add_argument('--language',      type=str,                required=True, help='Language of test document/the model was trained on')
    perp_parser.add_argument('--n',             type=check_ngram_length, required=True, help='N-gram length')
    perp_parser.add_argument('--format',        type=str,                help='Format used to store the model', choices={'normal', 'numpy'}, default='numpy')

    args = parser.parse_args()

    return args