import torch
import torch.nn as nn
from torch.utils import data
import torchtext.vocab as vocab
import nltk
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import jieba
import re
from gensim.models import KeyedVectors
import fasttext as ft
from spellchecker import spellchecker

# nltk.download('punkt')
# nltk.download('stopwords')
# tokenizer = RegexpTokenizer(r'\w+')

# print([key for key in vocab.pretrained_aliases.keys() if "glove" in key])
# cache_dir = "../glove/300"
# glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)

# wv_from_text = KeyedVectors.load('../dataset/tencent/baidubaike', mmap='r')
wv_from_text = KeyedVectors.load('../dataset/tencent/fasttext-crawl-300d-2M', mmap='r')
print('load word2vec finished')
sc = spellchecker.SpellChecker()

'''
def get_embeddings(sentence, sentence_len=64):
    try:
        # sentence = sentence.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        # print(tokens)
        return glove.get_vecs_by_tokens(tokens)
    except Exception as e:
        print(e, sentence)
        return torch.zeros(sentence_len, 300)
'''


def relation_to_index(relation):
    if relation == 'contradiction':
        return 0
    if relation == 'neutral':
        return 1
    if relation == 'entailment':
        return 2


def get_embeddings(sentence, sentence_len=64):
    try:
        remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
        string = re.sub(remove_chars, " ", sentence)
        words = ft.tokenize(string)
        li = []
        for word in words:
            word = word.lower()
            try:
                vc = wv_from_text.get_vector(word)
                li.append(vc)
            except Exception as e:
                correct_word = sc.correction(word)
                try:
                    vc = wv_from_text.get_vector(correct_word)
                    li.append(vc)
                    print('correction', word, correct_word)
                except Exception as e:
                    li.append([0.0 for _ in range(0, 300)])
                    print('instead zero', word)

        ten = torch.tensor(li)
        return ten
    except Exception as e:
        print('error', e, sentence)
        return torch.zeros(sentence_len, 300)


class MyDataset(data.Dataset):
    def __init__(self, tsv_path, sentence_len=64):
        self.reader = pd.read_csv(tsv_path,
                                  sep='\t',
                                  header=None,
                                  error_bad_lines=True)
        self.sentence1 = self.reader.iloc[:, 1]
        self.sentence2 = self.reader.iloc[:, 2]
        self.relation = self.reader.iloc[:, 0]
        self.len = len(self.relation)
        self.sentence_len = sentence_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        s1_embedding = get_embeddings(self.sentence1[index], self.sentence_len)
        s1_embedding, s1_mask = self.zero_padding(s1_embedding, target_len=self.sentence_len)

        s2_embedding = get_embeddings(self.sentence2[index], self.sentence_len)
        s2_embedding, s2_mask = self.zero_padding(s2_embedding, target_len=self.sentence_len)

        return relation_to_index(self.relation[index]), s1_embedding, s1_mask, s2_embedding, s2_mask

    def zero_padding(self, embedding, target_len):
        dim1 = embedding.shape[0]
        dim2 = embedding.shape[1]
        if dim1 >= target_len:
            embedding = embedding[0:target_len, :]
            mask = torch.ones_like(embedding)
            return embedding, mask
        else:
            mask = torch.ones_like(embedding)
            zeros = torch.zeros(target_len - dim1, dim2)
            mask = torch.cat((mask, zeros))
            # neg_inf = zeros - float('inf')
            embedding = torch.cat((embedding, zeros))
            return embedding, mask
