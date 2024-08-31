import numpy as np
<<<<<<< HEAD
import torch
import torch.nn as nn
from math import sqrt

from utils.masking import TriangularCausalMask


class FullAttention(nn.Module):
=======
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.ops import operations as P
from math import sqrt
from utils.masking import TriangularCausalMask

class FullAttention(nn.Cell):
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
<<<<<<< HEAD
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
=======
        self.dropout = nn.Dropout(keep_prob=1 - attention_dropout)

    def construct(self, queries, keys, values, attn_mask, tau=None, delta=None):
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

<<<<<<< HEAD
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
=======
        scores = P.MatMul()(queries, keys.transpose(0, 2, 1))
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

<<<<<<< HEAD
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
=======
            scores = scores - attn_mask.mask * np.inf

        A = self.dropout(P.Softmax(axis=-1)(scale * scores))
        V = P.MatMul()(A, values)

        if self.output_attention:
            return V.view(B, L, -1), A
        else:
            return V.view(B, L, -1), None

class AttentionLayer(nn.Cell):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
<<<<<<< HEAD
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
=======
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def construct(self, queries, keys, values, attn_mask, tau=None, delta=None):
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

<<<<<<< HEAD
        return self.out_projection(out), attn
=======
        return self.out_projection(out), attn
>>>>>>> 15d50d09666c0f1820500907f6e1a55b4753574c
