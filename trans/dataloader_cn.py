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

# nltk.download('punkt')
# nltk.download('stopwords')
# tokenizer = RegexpTokenizer(r'\w+')

# print([key for key in vocab.pretrained_aliases.keys() if "glove" in key])
# cache_dir = "../glove/300"
# glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)

wv_from_text = KeyedVectors.load('../dataset/tencent/baidubaike', mmap='r')
print('load word2vec finished')

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


def get_embeddings(sentence, sentence_len=64):
    try:
        remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
        string = re.sub(remove_chars, "", sentence)
        # l = jieba.cut(string)
        li = []
        for word in string:
            vc = wv_from_text.get_vector(word)
            li.append(vc)
        ten = torch.tensor(li)
        return ten
    except Exception as e:
        print(e, sentence)
        return torch.zeros(sentence_len, 300)


class MyDataset(data.Dataset):
    def __init__(self, tsv_path, sentence_len=64):
        self.reader = pd.read_csv(tsv_path,
                                  sep='\t',
                                  header=None,
                                  error_bad_lines=True)
        self.sentence1 = self.reader.iloc[:, 0]
        self.sentence2 = self.reader.iloc[:, 1]
        self.relation = self.reader.iloc[:, 2]
        self.len = len(self.relation)
        self.sentence_len = sentence_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        s1_embedding = get_embeddings(self.sentence1[index], self.sentence_len)
        s1_embedding = self.zero_padding(s1_embedding, target_len=self.sentence_len)
        s2_embedding = get_embeddings(self.sentence2[index], self.sentence_len)
        s2_embedding = self.zero_padding(s2_embedding, target_len=self.sentence_len)
        return self.relation[index], s1_embedding, s2_embedding

    def zero_padding(self, embedding, target_len):
        dim1 = embedding.shape[0]
        dim2 = embedding.shape[1]
        if dim1 >= target_len:
            return embedding[0:target_len, :]
        else:
            zeros = torch.zeros(target_len - dim1, dim2)
            # neg_inf = zeros - float('inf')
            embedding = torch.cat((embedding, zeros))
            return embedding
