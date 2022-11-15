# Models

Classes implemented are:

- `WordEmbedddings`: for handling pretrained word embeddings
- `BiLM`: Bidirectional Language Model using stacked bidirectional LSTMs. Stacking is done with skip connections and not concatenating the outputs of LSTMs when stacking. This means the input of the next forward LSTM is the output of the previous forward LSTM, same for backward LSTMs. This is different from stacked bidirectional LSTMs where the input of next bidirectional LSTM is the concatenation of the output of the previous forward and backward LSTMs.
- `ElMO`: for creating ELMO embeddings
- `Classifier`: Example of a downstream task that classifies Yelp reviews into 5 classes using the ElMO embeddings.

# Files

- `main.py` has all relevant classes and functions
- `clean.py` has one helper function to clean the data

# References

1. [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)
2. AllenNLP implementation

   - https://github.com/allenai/allennlp/blob/main/allennlp/modules/elmo.py#41
   - https://github.com/allenai/allennlp/blob/main/allennlp/modules/elmo.py#478
   - https://github.com/allenai/allennlp/blob/main/allennlp/modules/elmo_lstm.py#L21
   - https://github.com/allenai/allennlp/blob/main/allennlp/modules/scalar_mix.py#L10

3. https://jalammar.github.io/illustrated-bert/
