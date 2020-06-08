On using Hierarchical Contextual Document Embedding using BERT for Long Text Regression on MD&A disclosure
===============

Code and data for "On using Hierarchical Contextual Document Embedding using BERT for Long Text Regression on MD&A disclosure"

## Prerequisites
This code is written in python. To use it you will need:
- Python 3.6
- PyTorch [PyTorch](https://pytorch.org/)
- A recent version of [scikit-learn](https://scikit-learn.org/)
- A recent version of [Numpy](http://www.numpy.org)
- A recent version of [NLTK](http://www.nltk.org)
- [Tensorflow = 1.15.2](https://www.tensorflow.org)

## Getting started
We provide all the document embeddings [sum, max, mean, mean_max, max_mean, concat] in X_sum.pt, X_max.pt, X_mean_max.pt, X_max_mean.pt, X_concat.pt

We also provide the script create_emb.py to extract the above embeddings.

Additionally, we used extract_all_layers.py on our text documents to extract all 12 layers' hidden weights.
