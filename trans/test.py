import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from datetime import datetime

import dataloader_pawsx as dataloader
import model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_size = 300
output_size = 300
num_chans = [300, 300, 300, 300]
k_size = 2
dropout = 0.0
emb_dropout = 0.1
tied = False
sentence_len = 40

net = torch.load('./test-result/ca-3layer/net.m')
net.to(device)
loss = nn.CrossEntropyLoss()


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(net))

with torch.no_grad():
    test_loader = data.DataLoader(dataloader.MyDataset('../dataset/pawsx/test.tsv', sentence_len=sentence_len),
                                  batch_size=128,
                                  shuffle=False)

    index = 0
    correct = 0
    sum_test = 0

    for r, s1, s2 in test_loader:
        r = r.to(device)
        s1 = s1.to(device)
        s2 = s2.to(device)
        out = net(s1, s2)

        l = loss(out, r).sum()

        out = torch.argmax(out, dim=1)

        for i in range(out.shape[0]):
            sum_test += 1
            if out[i] == r[i]:
                correct += 1

    accuracy = correct / sum_test
    print('test', accuracy, datetime.now())
