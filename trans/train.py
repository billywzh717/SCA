import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from datetime import datetime

import dataloader, model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_size = 300
output_size = 300
num_chans = [300, 300, 300, 300]
k_size = 2
dropout = 0.3
emb_dropout = 0.1
tied = False
sentence_len = 40

net = model.MyNet(max_seq_len=sentence_len,
                  num_layers=6,
                  in_feature=300,
                  out_feature=300,
                  num_heads=4,
                  ffn_dim=2048,
                  dropout=dropout)
net.to(device)

train_loader = data.DataLoader(dataloader.MyDataset('../dataset/pair_train_correct.tsv', sentence_len=sentence_len),
                               batch_size=32,
                               shuffle=True)

num_epoch = 20
optimizer = optim.Adam(net.parameters(), lr=0.00001)

# print(net)

loss = nn.CrossEntropyLoss()

for epoch in range(num_epoch):
    index = 0
    for r, s1, s2 in train_loader:
        r = r.to(device)
        s1 = s1.to(device)
        s2 = s2.to(device)

        out = net(s1, s2)

        pre = torch.argmax(out, dim=1)
        correct = 0
        for i in range(out.shape[0]):
            if pre[i] == r[i]:
                correct += 1
        acc = correct / out.shape[0]

        l = loss(out, r).mean()

        index += 1
        if index % 10 == 0:
            print(epoch, index, l.item(), acc, datetime.now())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    torch.save(net, './model/net.m')

    with torch.no_grad():
        test_loader = data.DataLoader(dataloader.MyDataset('../dataset/pair_test.tsv', sentence_len=sentence_len),
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



