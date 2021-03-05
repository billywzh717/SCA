import torch
import torch.nn as nn

import cross_attention_encoder as cae


class TextCNN(nn.Module):
    def __init__(self, in_channels, out_channels, in_feature):
        super(TextCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(2, in_feature))

        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(4, in_feature))

        self.conv3 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(8, in_feature))

        self.conv4 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(16, in_feature))

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        input = x.unsqueeze(dim=1)
        conv1 = torch.max(self.conv1(input), dim=2)[0]
        conv2 = torch.max(self.conv2(input), dim=2)[0]
        conv3 = torch.max(self.conv3(input), dim=2)[0]
        conv4 = torch.max(self.conv4(input), dim=2)[0]
        output = self.flatten(torch.cat((conv1, conv2, conv3, conv4), dim=1))
        return output


class QCANet(nn.Module):

    def __init__(self,
                 max_seq_len=512,
                 num_layers=6,
                 in_feature=512,
                 out_feature=256,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        super(QCANet, self).__init__()

        self.cross_attention_layer = cae.CrossAttentionEncoderLayer(in_feature=in_feature,
                                                                    out_feature=out_feature,
                                                                    num_heads=num_heads,
                                                                    ffn_dim=ffn_dim,
                                                                    dropout=dropout)
        self.encoder_layers = nn.ModuleList(
            [cae.CrossAttentionEncoderLayer(in_feature=in_feature, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout) for _ in
             range(num_layers)])

        # self.seq_embedding = nn.Embedding(vocab_size + 1, in_feature, padding_idx=0)
        self.pos_embedding = cae.PositionalEncoding(in_feature, max_seq_len)

    def forward(self, q, k, v, attention_mask=None):
        q = self.pos_embedding(q)
        k = self.pos_embedding(k)
        v = self.pos_embedding(v)

        output, attention = self.cross_attention_layer(q, k, v)

        # self_attention_mask = cae.padding_mask(inputs, inputs)

        attentions = [attention]
        for encoder in self.encoder_layers:
            output, attention = encoder(output, output, q, attention_mask)
            attentions.append(attention)

        return output, attentions


class MyNet(nn.Module):
    def __init__(self,
                 max_seq_len=64,
                 num_layers=6,
                 in_feature=300,
                 out_feature=300,
                 num_heads=4,
                 ffn_dim=2048,
                 dropout=0.3):
        super(MyNet, self).__init__()
        self.qca = QCANet(max_seq_len=max_seq_len,
                          num_layers=num_layers,
                          in_feature=in_feature,
                          out_feature=out_feature,
                          num_heads=num_heads,
                          ffn_dim=ffn_dim,
                          dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=600, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(in_features=256, out_features=32),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=32),
            nn.Linear(in_features=32, out_features=3)
        )

    def forward(self, s1, s2):
        y1, y1_attentions = self.qca(s1, s2, s1)
        y2, y2_attentions = self.qca(s2, s1, s2)

        y1 = torch.mean(y1, dim=1)
        y2 = torch.mean(y2, dim=1)

        y = torch.cat((y1, y2), dim=1)
        # y = torch.cat((y, torch.sub(y1, y2)), dim=1)
        # y = torch.cat((y, torch.add(y1, y2)), dim=1)
        output = self.mlp(y)
        return output


if __name__ == '__main__':
    net = QCANet()
    print(net)
