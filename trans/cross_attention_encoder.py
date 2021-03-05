import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


def padding_mask(seq_k, seq_q):
    # seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
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

    def __init__(self, in_feature=768, out_feature=768, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.in_feature = in_feature
        self.dim_per_head = out_feature // num_heads
        self.num_heads = num_heads
        self.out_feature = out_feature

        self.linear_q = nn.Linear(in_feature, num_heads * out_feature)
        torch.nn.init.xavier_uniform_(self.linear_q.weight, gain=1)
        self.linear_k = nn.Linear(in_feature, num_heads * out_feature)
        torch.nn.init.xavier_uniform_(self.linear_k.weight, gain=1)
        self.linear_v = nn.Linear(in_feature, num_heads * out_feature)
        torch.nn.init.xavier_uniform_(self.linear_v.weight, gain=1)

        self.w = torch.nn.Parameter(
            torch.tensor(np.random.randn(in_feature, num_heads * out_feature), dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.w, gain=1)

        '''
        self.wq = torch.nn.Parameter(torch.tensor(np.random.randn(in_feature, num_heads*out_feature), dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.wq, gain=1)
        self.wk = torch.nn.Parameter(torch.tensor(np.random.randn(in_feature, num_heads*out_feature), dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.wk, gain=1)
        self.wv = torch.nn.Parameter(torch.tensor(np.random.randn(in_feature, num_heads*out_feature), dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.wv, gain=1)
        '''
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(num_heads * out_feature, in_feature)
        torch.nn.init.xavier_uniform_(self.linear_final.weight, gain=1)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(in_feature)

    def forward(self, query, key, value, attn_mask=None):
        # 残差连接
        residual = value

        dim_per_head = self.out_feature
        num_heads = self.num_heads
        batch_size = key.shape[0]

        # linear projection

        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)
        '''
        query = torch.matmul(query, self.wq)
        key = torch.matmul(key, self.wk)
        value = torch.matmul(value, self.wv)
        '''
        # split by heads
        query = query.view(batch_size, -1, num_heads, self.out_feature)
        key = key.view(batch_size, -1, num_heads, self.out_feature)
        value = value.view(batch_size, -1, num_heads, self.out_feature)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # scaled dot product attention
        context, attention = self.attention(query, key, value, self.dim_per_head, None)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.out_feature * self.num_heads)

        '''
        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        '''

        # final linear projection
        output = self.linear_final(context)
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention

    def attention(self, q, k, v, d_k, mask=None):
        attention = torch.matmul(q, k.transpose(-2, -1))
        scores = attention / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        output = torch.matmul(scores, v)
        return output, attention


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        pos_emb = self.pe[:, :x.size(1)]
        x = x + pos_emb
        return x


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


class CrossAttentionEncoderLayer(nn.Module):
    """Encoder的一层。"""

    def __init__(self, in_feature=768, out_feature=256, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(CrossAttentionEncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(in_feature=in_feature,
                                            out_feature=out_feature,
                                            num_heads=num_heads,
                                            dropout=dropout)
        self.feed_forward = PositionalWiseFeedForward(in_feature, ffn_dim, dropout)

    def forward(self, q, k, v, attn_mask=None):
        # self attention
        # context, attention = self.attention(q, k, v, padding_mask)
        context, attention = self.attention(q, k, v)
        # feed forward network
        output = self.feed_forward(context)

        return output, attention
