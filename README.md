# ANLP - Coursework 1
This repository contains code for training and using an ngram language model as per the requirements of Assignment 1 for the Accelerated Natural Language Processing course (ANLP) at the University of Edinburgh.

## Usage
A 'data' folder is assumed to exist under the current working directory.
This folder be used to store training data (text files consisting of sentences in a particular language) as well as models

If the format chosen is 'numpy', the program reads from/writes to model file `data/model-vec.<language>.<n>.npz`.
If the format is 'normal',  the program reads from/writes to model file `data/model-display.<language>.<n>`.

### Training
```bash
>>> python main_final.py train -h
usage: main_final.py train [-h] --training_file TRAINING_FILE --language
                           LANGUAGE [--train_type {interpolation,add_alpha}]
                           --n N [--format {normal,numpy}]

optional arguments:
  -h, --help            show this help message and exit
  --training_file TRAINING_FILE
                        Name of file/document to train the model with
  --language LANGUAGE   Language of training document
  --train_type {interpolation,add_alpha}
                        Specify how to train the model (add_alpha,
                        interpolation)
  --n N                 N-gram length
  --format {normal,numpy}
                        Whether to save model as human-readable or npz
```

The probability matrix (i.e. the model) will be saved in `data/model-vec.<language>.<n>.npz` to exist (if the chosen format is `numpy`), or `data/model-display.<language>.<n>` (if the chosen format is `normal`).

#### Example
```bash
python main_final.py train --training_file data/training.en --language en --train_type interpolation --n 3 --format numpy
```

This will save the model in `data/model-vec.en.3.npz`.

### Generating
```bash
>>> python3.6 main_final.py generate -h
usage: main_final.py generate [-h] --language LANGUAGE --n N
                              [--format {numpy,normal}]

optional arguments:
  -h, --help            show this help message and exit
  --language LANGUAGE   Language the model was trained on
  --n N                 N-gram length
  --format {numpy,normal}
                        Format used to store the model
```

This will require the model file `data/model-vec.<language>.<n>.npz` (if the chosen format is `numpy`), or `data/model-display.<language>.<n>` (if the chosen format is `normal`) to exist.

#### Example
```bash
python main_final.py generate --language en --n 3 --format numpy
```

This will read the model from `data/model-vec.en.3.npz` and generate a sequence.

### Calculating perplexity
```bash
>>> python3.6 main_final.py perp -h
usage: main_final.py perp [-h] --document_file DOCUMENT_FILE --language
                          LANGUAGE --n N [--format {normal,numpy}]

optional arguments:
  -h, --help            show this help message and exit
  --document_file DOCUMENT_FILE
                        Name of test file/document to calculate the perplexity
                        of the model
  --language LANGUAGE   Language of test document/the model was trained on
  --n N                 N-gram length
  --format {normal,numpy}
                        Format used to store the model
```

The document to specify the path of is any document to calculate the perplexity of in the given language.
This will require the model file `data/model-vec.<language>.<n>.npz` (if the chosen format is `numpy`), or `data/model-display.<language>.<n>` (if the chosen format is `normal`) to exist.

#### Example
```bash
python main_final.py perp --document_file data/test --language en --n 3 --format numpy
```

This will read the file `data/test`, preprocess it line-by-line, and calculate its perplexity on the model stored at `data/model-vec.en.3.npz`.