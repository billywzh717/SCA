import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from datetime import datetime

import dataloader_cn as dataloader
import cat_net_model as model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(net, data_path):
    with torch.no_grad():
        test_loader = data.DataLoader(
            dataloader.MyDataset(data_path, sentence_len=sentence_len),
            batch_size=32,
            shuffle=False,
            drop_last=True)

        correct = 0
        sum_test = 0

        for r, s1, s1_mask, s2, s2_mask in test_loader:
            r = r.to(device)
            s1 = s1.to(device)
            s1_mask = s1_mask.to(device)
            s2 = s2.to(device)
            s2_mask = s2_mask.to(device)

            # l, acc = net(s1, s2, r)

            out = net(s1, s2, r, s1_mask=None, s2_mask=None)

            '''
            sum_test += 1
            l, acc = net(s1, s2, r)
            correct += acc
            '''

            out = torch.argmax(out, dim=1)
            for i in range(out.shape[0]):
                sum_test += 1
                if out[i] == r[i]:
                    correct += 1

        accuracy = correct / sum_test
        print('test', accuracy, datetime.now())


input_features = 300
output_features = 300
num_epoch = 100
num_layers = 1
num_head = 4
weight_decay = 0.0

lr = 0.00001
dropout = 0.3
num_blocks = 6
sentence_len = 30
batch_size = 32
num_classes = 2

# data_path = '../dataset/snli/train_no_stop_word.tsv'
# test_data_path = '../dataset/snli/test_no_stop_word.tsv'

# data_path = '../dataset/snli/neutral_entailment.tsv'
# test_data_path = '../dataset/snli/neutral_entailment_test.tsv'

# data_path = '../dataset/snli/contradiction_entailment.tsv'
# test_data_path = '../dataset/snli/contradiction_entailment_test.tsv'

data_path = '../dataset/lcqmc-clean/train.tsv'
test_data_path = '../dataset/lcqmc-clean/test.tsv'

# data_path = '../dataset/pawsx/train.tsv'

'''
net = model.MyNet(sentence_len=sentence_len,
                  num_layers=num_layers,
                  in_features=input_features,
                  out_features=output_features,
                  num_heads=4,
                  ffn_dim=2048,
                  dropout=dropout,
                  num_blocks=3)
'''
net = model.SiameseCatNet(in_features=input_features,
                          out_features=output_features,
                          num_blocks=num_blocks,
                          num_head=num_head,
                          dropout=dropout,
                          num_classes=num_classes)
net.to(device)

train_loader = data.DataLoader(dataloader.MyDataset(data_path, sentence_len=sentence_len),
                               batch_size=batch_size,
                               shuffle=True,
                               drop_last=True)

optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

# print(net)

# loss = nn.CrossEntropyLoss()
focal_loss = model.FocalLoss(num_classes=num_classes)

for epoch in range(num_epoch):
    index = 0
    for r, s1, s1_mask, s2, s2_mask in train_loader:
        r = r.to(device)
        s1 = s1.to(device)
        s1_mask = s1_mask.to(device)
        s2 = s2.to(device)
        s2_mask = s2_mask.to(device)

        # l, acc = net(s1, s2, r)

        out = net(s1, s2, r, s1_mask=None, s2_mask=None)
        # l = loss(out, r).mean()
        l = focal_loss(out, r)

        pre = torch.argmax(out, dim=1)
        correct = 0
        for i in range(out.shape[0]):
            if pre[i] == r[i]:
                correct += 1
        acc = correct / out.shape[0]

        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        index += 1
        if index % 10 == 0:
            print(epoch, index, l.item(), acc, datetime.now())
        if index % 500 == 0:
            test(net, test_data_path)

    torch.save(net, './model/final.ckp')
    print('epoch')
    test(net, test_data_path)
