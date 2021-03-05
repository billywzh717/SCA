import jieba
from gensim.models import KeyedVectors
from datetime import datetime
import torch

import re

wv_from_text = KeyedVectors.load_word2vec_format('dataset/tencent/sgns.baidubaike.bigram-char', binary=False)

def encoder(sentence):
    remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
    string = re.sub(remove_chars, "", sentence)
    l = jieba.cut(string)
    li = []
    for word in l:
        vc = wv_from_text.get_vector(word)
        li.append(vc)
    ten = torch.tensor(li)
    return ten

sentence = "+蚂=蚁！花!呗/期?免,息★.---《平凡的世界》：了*解一（#@）个“普通人”波涛汹涌的内心世界！"
ten = encoder(sentence)
print(ten)
