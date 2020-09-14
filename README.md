Hierarchical Contextual Document Embeddings for Long Financial Text Regression
===============

Code and data for "Hierarchical Contextual Document Embeddings for Long Financial Text Regression"

## Prerequisites
This code is written in python. To use it you will need:
- Python 3.6
- [PyTorch [PyTorch] = 1.2.0](https://pytorch.org/)
- A recent version of [scikit-learn](https://scikit-learn.org/)
- A recent version of [Numpy](http://www.numpy.org)
- A recent version of [NLTK](http://www.nltk.org)
- [Tensorflow = 1.15.2](https://www.tensorflow.org)

## Getting started
We provide all the document embeddings [sum, max, mean, mean_max, max_mean, concat] in ```data``` directory.

We also provide the script ```create_emb.py``` to extract the above embeddings.

Additionally, we used ```extract_all_layers.py``` on our text documents to extract all 12 layers' hidden weights.

```data_2436.csv``` contains tokenized text for different features such as ```words```, ```sentiment_words``` and ```syntactically_expanded_sentiment_words```. This data file can be found at [Zenodo](https://zenodo.org/record/4029239#.X1-r5HXNY5k) 
