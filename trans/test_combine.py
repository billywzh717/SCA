from datetime import datetime

import torch
from torch.utils import data

import dataloader as dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_size = 300
output_size = 300
num_chans = [300, 300, 300, 300]
k_size = 2
dropout = 0.0
emb_dropout = 0.1
tied = False
sentence_len = 32

n_e_net = torch.load('./model/n_e.ckp')
c_n_net = torch.load('./model/c_n.ckp')
c_e_net = torch.load('./model/c_e.ckp')
n_e_net.to(device)
c_n_net.to(device)
c_e_net.to(device)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

with torch.no_grad():
    test_loader = data.DataLoader(dataloader.MyDataset('../dataset/snli/pair_test.tsv', sentence_len=sentence_len),
                                  batch_size=32,
                                  shuffle=False)

    index = 0
    correct = 0
    sum_test = 0

    for r, s1, s2 in test_loader:
        r = r.to(device)
        s1 = s1.to(device)
        s2 = s2.to(device)
        c_n = c_n_net(s1, s2, r)
        n_e = n_e_net(s1, s2, r)
        c_e = c_e_net(s1, s2, r)

        c_n_label = torch.argmax(c_n, dim=1)
        n_e_label = torch.argmax(n_e, dim=1)
        c_e_label = torch.argmax(c_e, dim=1)

        answer = []
        for i in range(r.shape[0]):
            if c_n_label[i] == 0:
                if c_e_label[i] == 0:
                    answer.append(0)
                elif c_e_label[i] == 1:
                    answer.append(2)
                    '''
                    if n_e_label[i] == 0:
                        answer.append(1)
                    elif n_e_label[i] == 1:
                        answer.append(2)
                    '''
            if c_n_label[i] == 1:
                if n_e_label[i] == 0:
                    answer.append(1)
                elif n_e_label[i] == 1:
                    answer.append(2)
                    '''
                    if c_e_label[i] == 0:
                        answer.append(0)
                    elif c_e_label[i] == 1:
                        answer.append(2)
                    '''

        c_n_label -= 1
        n_e_label = n_e_label
        c_e_label = 2 * c_e_label - 1

        combine_label = c_n_label + n_e_label + c_e_label

        for i in range(r.shape[0]):
            if combine_label[i] < 0:
                combine_label[i] = 0
            elif combine_label[i] == 0:
                combine_label[i] = 1
            elif combine_label[i] > 0:
                combine_label[i] = 2

        '''
        for i in range(r.shape[0]):
            if c_n_label[i] == 0:
                combine_label.append(0)
            if c_n_label[i] == 1:
                if n_e_label[i] == 0:
                    combine_label.append(1)
                if n_e_label[i] == 1:
                    combine_label.append(2)
        '''

        for i in range(r.shape[0]):
            sum_test += 1
            if answer[i] == r[i]:
                correct += 1

    accuracy = correct / sum_test
    print('test', accuracy, datetime.now())
