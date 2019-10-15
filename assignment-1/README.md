## Overview
`main_final.py` consists of a simple command line interface to use this language model. 
Running `python main_final.py` shows the interface for each function. The `format` argument specified in which format the data should be stored. The `normal` option will be slower in writing the file, but the data will be in readable format.

The rest of the files provide the 'actual' functionality of the language model i.e. add alpha and interpolation.
## Usage
This assumes a 'data' folder exists under the current working directory.

If the format chosen is 'numpy', the program looks reads from/writes to model file `data/model-vec.<language>.<n>.npz`.

If the format is 'normal',  the program reads from/writes to model file `data/model-display.<language>.<n>`.
#### Training
```
python main.py train <path/to/training/file> <language> <train_type> <n> <format>
```
The probabilities will be saved in `data/model-vec.<language>.<n>.npz` to exist (if the chosen format is `numpy`), or `data/model-display.<language>.<n>` (if the chosen format is `normal`).
The training performs an 80/10/10 split into training, validation and test data. Using the validation data, it performs a grid search to find the optimal hyperparameters. After this, it reports the perplexity of the resulting model on the test set. Interpolation training uses the two smaller sets as separate validation sets, to first tune the alpha values, and then the lambda values. The perplexity is reported on the file `data/test`.
###### Example
```
python main_final.py train data/training.en en interpolation 5 numpy
```

This will save the model in `data/model-vec.en.5.npz`
#### Generating
```
python main.py generate <language> <n> <format>
```
This will require the probabilities file `data/model-vec.<language>.<n>.npz` to exist (if the chosen format is `numpy`), or `data/model-display.<language>.<n>` (if the chosen format is `normal`).
###### Example
```
python main_final.py generate en 5 numpy
```
This will read the model from `data/model-vec.en.5.npz` and generate a sequence.
#### Calculating perplexity
```
python main.py perp <path/to/document> <language> <n> <format>
```
The document to specify the path of is any document to calculate the perplexity of in the given language.
This will also require the probabilities file `data/model-vec.<language>.<n>.npz` to exist (if the chosen format is `numpy`), or `data/model-display.<language>.<n>` (if the chosen format is `normal`) .
###### Example
```
python main_final.py perp data/test en 5 numpy
```
This will read the file `data/test`, preprocess it line-by-line, calculate its perplexity on the model stored at `data/model-vec.en.5.npz` and report it.