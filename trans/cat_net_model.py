import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)
        # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class PositionEncoding(nn.Module):

    def __init__(self, in_features, max_len=5000):
        super(PositionEncoding, self).__init__()

        pe = torch.zeros(max_len, in_features)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_features, 2).float() * (-math.log(10000.0) / in_features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        pos_emb = self.pe[:, :x.size(1)]
        x = x + pos_emb
        return x


class BiLSTM(nn.Module):
    def __init__(self, in_features=300, hidden_size=300, dropout=0.0):
        super(BiLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_features)

        self.bi_lstm = nn.LSTM(batch_first=True,
                               bidirectional=True,
                               input_size=in_features,
                               hidden_size=hidden_size,
                               dropout=dropout)

    def forward(self, x):
        y, y_hn_cn = self.bi_lstm(x)
        hidden_size = y.shape[2]
        y_forward = y[:, :, :hidden_size // 2]
        y_backward = y[:, :, hidden_size // 2:]
        y = (y_forward + y_backward) / 2
        output = self.layer_norm(x + y)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.dim_per_head = in_features // num_heads

        self.linear_q = nn.Linear(in_features=in_features, out_features=num_heads * out_features)
        self.linear_k = nn.Linear(in_features=in_features, out_features=num_heads * out_features)
        self.linear_v = nn.Linear(in_features=in_features, out_features=num_heads * out_features)

        self.linear_final = nn.Linear(in_features=num_heads * out_features,
                                      out_features=in_features)
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, q, k, v, q_mask=None, k_mask=None, v_mask=None):
        residual = v
        batch_size = q.shape[0]

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        q = q.view(batch_size, -1, self.num_heads, self.out_features).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.out_features).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.out_features).transpose(1, 2)

        context, attention = self.dot_product_attention(q, k, v, self.dim_per_head, q_mask, k_mask, v_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.out_features * self.num_heads)

        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output, attention

    def dot_product_attention(self, q, k, v, dim, q_mask=None, k_mask=None, v_mask=None):
        # batch head sub_sentence embedding
        mask = None
        if (q_mask is not None) and (k_mask is not None):
            mask = torch.matmul(q_mask, k_mask.transpose(1, 2)) / 300
        attention = torch.matmul(q, k.transpose(-2, -1))
        scores = attention / math.sqrt(dim)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            scores = F.softmax(scores, dim=-1)
            scores = scores.masked_fill(mask == 0, 0)
        else:
            scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        return output, attention


class CatBlock(nn.Module):
    def __init__(self, in_features=768, out_features=768, num_heads=4, dropout=0.0):
        super(CatBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=in_features)
        self.multi_head_attention = MultiHeadAttention(in_features, out_features, num_heads, dropout)
        self.bilstm = BiLSTM(in_features=in_features, dropout=dropout)

    def forward(self, q, k, v, q_mask=None, k_mask=None, v_mask=None):
        output, attention = self.multi_head_attention(q, k, v, q_mask, k_mask, v_mask)
        output = self.bilstm(output)
        return output, attention


class SiameseCatNet(nn.Module):
    def __init__(self, in_features=300,
                 out_features=300,
                 num_blocks=12,
                 num_head=4,
                 dropout=0.0,
                 num_classes=2):
        super(SiameseCatNet, self).__init__()
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.cat_net1 = nn.ModuleList([CatBlock(in_features=in_features,
                                               out_features=out_features,
                                               num_heads=num_head,
                                               dropout=dropout) for _ in range(num_blocks)])
        self.cat_net2 = nn.ModuleList([CatBlock(in_features=in_features,
                                               out_features=out_features,
                                               num_heads=num_head,
                                               dropout=dropout) for _ in range(num_blocks)])
        self.position_encoding = PositionEncoding(in_features=in_features)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=2 * in_features, out_features=256),
            nn.ReLU(),
            # nn.BatchNorm1d(num_features=256),
            nn.LayerNorm(256),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            # nn.BatchNorm1d(num_features=128),
            nn.LayerNorm(128),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            # nn.BatchNorm1d(num_features=32),
            nn.LayerNorm(32),
            nn.Linear(in_features=32, out_features=num_classes)
        )

    def forward(self, s1, s2, r=None, s1_mask=None, s2_mask=None):
        s1 = self.position_encoding(s1)
        s2 = self.position_encoding(s2)
        s1_attentions = []
        s2_attentions = []
        for i in range(self.num_blocks):
            # s1_mask = None
            # s2_mask = None
            s1, s1_attention = self.cat_net1[i](s2, s1, s1, s2_mask, s1_mask, s1_mask)
            s1_attentions.append(s1_attention)
            s2, s2_attention = self.cat_net1[i](s1, s2, s2, s1_mask, s2_mask, s2_mask)
            s2_attentions.append(s2_attention)

        y1 = torch.mean(s1, dim=1)
        y2 = torch.mean(s2, dim=1)
        # add = torch.add(y1, y2)
        # sub = torch.sub(y1, y2)
        # mul = torch.dot(y1, y2)
        y = torch.cat((y1, y2), dim=1)
        output = self.mlp(y)
        return output
