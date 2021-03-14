import torch
import torch.nn as nn
from torch.utils import data
import torchtext.vocab as vocab
import nltk
from nltk.tokenize import RegexpTokenizer
import pandas as pd

# nltk.download('punkt')
# nltk.download('stopwords')
tokenizer = RegexpTokenizer(r'\w+')

# print([key for key in vocab.pretrained_aliases.keys() if "glove" in key])
cache_dir = "./glove/300"
glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)

print(glove.get_vecs_by_tokens(['ShoeGasm']))
