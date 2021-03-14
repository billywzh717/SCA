import urllib3
import base64
import torch
import torch.nn as nn
import torchtext.vocab as vocab
import nltk
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import jieba
import re
from gensim.models import KeyedVectors
import fasttext as ft
import spellchecker

# http = urllib3.PoolManager()
# r = http.request('GET', 'http://127.0.0.1:5000/word2vec/model?word=好')
# data = r.data
# print(data)
# bbs = str(base64.b64decode(data))
# print(bbs)

# wv_from_text = KeyedVectors.load_word2vec_format('./dataset/tencent/crawl-300d-2M.vec', binary=False)
# wv_from_text.save('./dataset/tencent/fasttext-crawl-300d-2M')

wv_from_text = KeyedVectors.load('./dataset/tencent/fasttext-crawl-300d-2M', mmap='r')
vc = wv_from_text.get_vector('graphittied')
print(vc)

# print(ft.tokenize('A person on a horse jumps over a broken down airplane.'))
# sc = spellchecker.spellchecker.SpellChecker()
# result = sc.correction('mudpuddles')
# print(result)


# def get_embeddings(sentence, sentence_len=64):
#     try:
#         remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
#         string = re.sub(remove_chars, "", sentence)
#         l = jieba.cut(string)
#         li = []
#         for word in l:
#             vc = wv_from_text.get_vector(word)
#             li.append(vc)
#         ten = torch.tensor(li)
#         return ten
#     except Exception as e:
#         print(e, sentence)
#         return torch.zeros(sentence_len, 300)
#
# ten = get_embeddings('好')
# print(ten)