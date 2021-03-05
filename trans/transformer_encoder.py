import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def padding_mask(seq_k, seq_q):
    # seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


class ScaledDotProductAttention(nn.Module):

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, in_feature=768, out_feature=256, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = in_feature // num_heads
        self.num_heads = num_heads
        self.out_feature = out_feature

        self.wq = torch.nn.Parameter(torch.tensor(np.random.randn(in_feature, out_feature), dtype=torch.float32))
        self.wk = torch.nn.Parameter(torch.tensor(np.random.randn(in_feature, out_feature), dtype=torch.float32))
        self.wv = torch.nn.Parameter(torch.tensor(np.random.randn(in_feature, out_feature), dtype=torch.float32))

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(in_feature, in_feature)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(in_feature)

    def forward(self, query, key, value, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.shape[0]

        # linear projection
        query = torch.matmul(query, self.wq)
        key = torch.matmul(key, self.wk)
        value = torch.matmul(value, self.wv)

        # split by heads
        key = key.view(batch_size * num_heads, -1, self.out_feature)
        value = value.view(batch_size * num_heads, -1, self.out_feature)
        query = query.view(batch_size * num_heads, -1, self.out_feature)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):

        super(PositionalEncoding, self).__init__()

        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, torch.from_numpy(position_encoding)))

        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):

        max_len = torch.max(input_len)
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = torch.LongTensor(
            [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, in_feature=768, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(in_feature, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, in_feature, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_feature)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class EncoderLayer(nn.Module):
    """Encoder的一层。"""

    def __init__(self, in_feature=768, out_feature=256, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(in_feature=in_feature,
                                            out_feature=out_feature,
                                            num_heads=num_heads,
                                            dropout=dropout)
        self.feed_forward = PositionalWiseFeedForward(in_feature=in_feature,
                                                      ffn_dim=ffn_dim,
                                                      dropout=dropout)

    def forward(self, q, k, v, attn_mask=None):
        # self attention
        context, attention = self.attention(q, k, v)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):

    def __init__(self,
                 max_seq_len,
                 num_layers=6,
                 in_feature=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(in_feature=in_feature, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout) for _ in
             range(num_layers)])

        self.pos_embedding = PositionalEncoding(in_feature, max_seq_len)

    def forward(self, inputs, inputs_len):
        output = inputs + self.pos_embedding(inputs_len)

        self_attention_mask = padding_mask(inputs, inputs)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions


class TransformerEncoder(nn.Module):

    def __init__(self,
                 src_max_len=512,
                 num_layers=6,
                 in_feature=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.2):
        super(TransformerEncoder, self).__init__()

        self.encoder = Encoder(max_seq_len=src_max_len,
                               num_layers=num_layers,
                               in_feature=in_feature,
                               num_heads=num_heads,
                               ffn_dim=ffn_dim,
                               dropout=dropout)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_len):
        output, enc_self_attn = self.encoder(src_seq, src_len)
        return output, enc_self_attn


if __name__ == '__main__':
    net = Encoder(512)
    print(net)
