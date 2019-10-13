## Usage
#### Training
```
python main.py train <path/to/training/file> <language> <train_type> <n> <format>
```
The probabilities will be saved in `data/model.<language>`.
The training performs an 80/10/10 split into training, validation and test data. Using the validation data, it performs a grid search to find the optimal hyperparameters. After this, it reports the perplexity of the resulting model on the test set.
#### Generating
```
python main.py generate <language> <n> <format>
```
This will require the probabilities file `data/model.<language>` to exist.
#### Calculating perplexity
```
python main.py perp <path/to/document> <language> <n> <format>
```
The document to specify the path of is any document to calculate the perplexity of in the given language.
This will also require the probabilities file `data/model.<language>` to exist.
