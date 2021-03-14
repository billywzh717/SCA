import torch
import torch.nn as nn
import torch.nn.functional as F

import cross_attention_encoder as cae


class BiGRU(nn.Module):
    def __init__(self, in_features=300, sentence_len=40):
        super(BiGRU, self).__init__()
        self.forward_gru = nn.GRU(input_size=in_features, hidden_size=sentence_len, bidirectional=True)

    def forward(self, x):
        forward_hiddens, = self.forward_gru(x)
        backward_hidden = self.backward_gru(x)
        pass


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        flatten = self.flatten(pool3)
        return flatten


class TextCNN(nn.Module):
    def __init__(self, in_channels, out_channels, in_features):
        super(TextCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1, in_features))

        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(2, in_features))

        self.conv3 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, in_features))

        self.conv4 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(4, in_features))

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        input = x.unsqueeze(dim=1)

        conv1 = self.conv1(input)
        conv2 = self.conv2(input)
        conv3 = self.conv3(input)
        conv4 = self.conv4(input)

        '''
        conv1 = torch.max(self.conv1(input), dim=2)[0]
        conv2 = torch.max(self.conv2(input), dim=2)[0]
        conv3 = torch.max(self.conv3(input), dim=2)[0]
        conv4 = torch.max(self.conv4(input), dim=2)[0]
        '''

        output = self.flatten(torch.cat((conv1, conv2, conv3, conv4), dim=2))
        return output


class QCANet(nn.Module):

    def __init__(self,
                 sentence_len=512,
                 num_layers=6,
                 in_features=512,
                 out_features=256,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        super(QCANet, self).__init__()

        self.cross_attention_layer = cae.CrossAttentionEncoderLayer(in_features=in_features,
                                                                    out_features=out_features,
                                                                    num_heads=num_heads,
                                                                    ffn_dim=ffn_dim,
                                                                    dropout=dropout)
        self.encoder_layers = nn.ModuleList(
            [cae.CrossAttentionEncoderLayer(in_features=in_features, num_heads=num_heads, ffn_dim=ffn_dim,
                                            dropout=dropout) for _ in
             range(num_layers)])

        # self.seq_embedding = nn.Embedding(vocab_size + 1, in_features, padding_idx=0)
        self.pos_embedding = cae.PositionalEncoding(in_features, sentence_len)

    def forward(self, q, k, v, attention_mask=None):
        q = self.pos_embedding(q)
        k = self.pos_embedding(k)
        v = self.pos_embedding(v)

        output, attention = self.cross_attention_layer(q, k, v)

        # self_attention_mask = cae.padding_mask(inputs, inputs)

        attentions = [attention]
        for encoder in self.encoder_layers:
            output, attention = encoder(output, output, output, attention_mask)
            attentions.append(attention)

        return output, attentions


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        # self.margin = torch.tensor(margin, requires_grad=True)
        self.margin = 1.0

    def forward(self, x1, x2, label):
        dis = torch.sum(torch.square(x1 - x2), dim=1)
        temp2 = torch.clamp(self.margin - dis, min=0.0)
        loss = label * dis + (1 - label) * temp2
        loss = 0.5 * loss.mean()

        correct = 0
        for i in range(dis.shape[0]):
            if (dis[i] <= self.margin and label[i] == 1) or (dis[i] > self.margin and label[i] == 0):
                correct += 1

        acc = correct / dis.shape[0]

        return loss, acc


class Block(nn.Module):
    def __init__(self,
                 sentence_len=64,
                 num_layers=6,
                 in_features=300,
                 out_features=300,
                 num_heads=4,
                 ffn_dim=2048,
                 dropout=0.3):
        super(Block, self).__init__()
        self.bi_lstm = nn.LSTM(batch_first=True,
                               bidirectional=True,
                               input_size=in_features,
                               hidden_size=in_features)
        self.qca = QCANet(sentence_len=sentence_len,
                          num_layers=num_layers,
                          in_features=in_features,
                          out_features=out_features,
                          num_heads=num_heads,
                          ffn_dim=ffn_dim,
                          dropout=dropout)
        self.textcnn = TextCNN(in_channels=1, out_channels=1, in_features=in_features)
        self.cnn = CNN(in_channels=1, out_channels=1)
        self.normal = nn.LayerNorm(normalized_shape=300)

    def forward(self, s1, s2, r):
        # attention
        y1, y1_attentions = self.qca(s1, s2, s1)
        y2, y2_attentions = self.qca(s2, s1, s2)
        # bi-lstm
        y1, y1_hn_cn = self.bi_lstm(y1)
        y2, y2_hn_cn = self.bi_lstm(y2)
        hidden_size = y1.shape[2]
        y1_forward = y1[:, :, :hidden_size // 2]
        y1_backward = y1[:, :, hidden_size // 2:]
        y1 = (y1_forward + y1_backward) / 2
        y2_forward = y2[:, :, :hidden_size // 2]
        y2_backward = y2[:, :, hidden_size // 2:]
        y2 = (y2_forward + y2_backward) / 2
        # residual
        y1 += s1
        y2 += s2
        return y1, y2, r


class MyNet(nn.Module):
    def __init__(self,
                 sentence_len=64,
                 num_layers=6,
                 in_features=300,
                 out_features=300,
                 num_heads=4,
                 ffn_dim=2048,
                 dropout=0.3,
                 num_blocks=3):
        super(MyNet, self).__init__()
        self.textcnn = TextCNN(in_channels=1, out_channels=1, in_features=in_features)
        self.cnn = CNN(in_channels=1, out_channels=1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=4 * 300 + 1, out_features=256),
            nn.ReLU(),
            # nn.BatchNorm1d(num_features=256),
            nn.Linear(in_features=256, out_features=32),
            nn.ReLU(),
            # nn.BatchNorm1d(num_features=32),
            nn.Linear(in_features=32, out_features=2)
        )
        # self.cosine = nn.CosineEmbeddingLoss()
        self.loss = ContrastiveLoss()
        self.normal = nn.LayerNorm(normalized_shape=300)

        self.blocks = nn.ModuleList([Block(sentence_len=sentence_len,
                                           num_layers=num_layers,
                                           in_features=in_features,
                                           out_features=out_features,
                                           num_heads=num_heads,
                                           ffn_dim=ffn_dim,
                                           dropout=dropout) for _ in range(num_blocks)])

    def forward(self, s1, s2, r):

        for block in self.blocks:
            s1, s2, r = block(s1, s2, r)

        # y1 = self.textcnn(s1)
        # y2 = self.textcnn(s2)


        # y1 = self.cnn(s1)
        # y2 = self.cnn(s2)

        # loss, acc = self.loss(self.normal(y1), self.normal(y2), r)

        y1 = torch.mean(s1, dim=1)
        y2 = torch.mean(s2, dim=1)
        add = torch.add(y1, y2)
        sub = torch.sub(y1, y2)
        mul = torch.matmul(y1.unsqueeze(1), y2.unsqueeze(1).transpose(1, 2)).squeeze(1)

        y = torch.cat((y1, add, sub, mul, y2), dim=1)
        output = self.mlp(y)

        return output


if __name__ == '__main__':
    net = QCANet()
    print(net)
