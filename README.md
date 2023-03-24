# 4490Z-thesis
Codebase for my undergraduate thesis project

# Running the code
## Overview
experiment.py contains the code to feed the neutral and biased mediating triplets to the contextual models. It also runs the vector diversity experiment.

experiment2.py contains the code to feed the neutral and biased mediating triplets into the non-contextual model.

## Libraries
Install these following libraries to run experiment.py:
- import torch
- from transformers import BertTokenizer, BertModel
- import numpy as np
- from scipy.spatial.distance import cosine
- import csv

Install these additional libraries to run experiment2.py:
- import torchtext
- import gensim.downloader

## Donwloading models
experiment.py will download the BERT-base, MedBERT, and MentalBERT models locally.
experiment2.py will download the GloVe, fasttext, and word2vec models locally.

This will consume a few gigabyes in memory.

# Changelog
- March 24, 2023 - uploaded the python scripts
- March 13, 2023 - uploaded the raw data from the experiment output
