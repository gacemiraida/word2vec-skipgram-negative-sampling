# Word2Vec: Skip-Gram with Negative Sampling

A practical implementation of Word2Vec using the Skip-Gram architecture and Negative Sampling with PyTorch.

## Overview

This project explores how neural networks learn distributed word representations from text using the Skip-Gram model.

The implementation uses the Penn Treebank (PTB) dataset and covers text preprocessing, context generation, negative sampling, model training, and word similarity evaluation.

## Objectives

- Understand word embeddings and the Skip-Gram architecture.
- Implement subsampling of frequent words.
- Generate context words using a sliding window.
- Apply negative sampling.
- Train Word2Vec using PyTorch.
- Find semantically similar words using cosine similarity.
- Experiment with dynamic context and negative sampling.

## Technologies

- Python
- PyTorch
- Jupyter Notebook
- D2L
- Natural Language Processing

## Project Structure

```text
word2vec-skipgram-negative-sampling/
│
├── README.md
└── Word2Vec_SkipGram.ipynb
```

## Implementation

The notebook covers:

1. Dataset loading and vocabulary construction.
2. Subsampling frequent words.
3. Center and context generation.
4. Negative sampling.
5. Minibatch construction.
6. Skip-Gram model and embedding layers.
7. Binary cross-entropy loss.
8. Model training.
9. Cosine similarity for finding similar words.
10. Dynamic sampling extension.

## Experiments

The project explores the effect of hyperparameters and compares fixed sampling with dynamic sampling.

## Learning Outcomes

Through this project, I explored the fundamentals of Word2Vec, embedding representations, negative sampling, and neural network training for Natural Language Processing.

## Acknowledgements

This project is based on educational material from the Dive into Deep Learning (D2L) book, with additional experiments and implementation work.

## Author

**Raida Gacemi**
