from math import sqrt
import numpy as np

import torch
import torch.nn as nn
from einops import repeat

from layers.Attn_Bias import BinaryAttentionBias
from utils.masking import TriangularCausalMask, TimerMultivariateMask, TimerMultivariateWithFutureCovariateMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
        

class TimeAttention(nn.Module):
    def __init__(self, mask_flag=True, binary_bias=False, d_model=512, num_heads=8, covariate=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(TimeAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.binary_bias = binary_bias
        self.covariate = covariate
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        if self.binary_bias:
            self.attn_bias = BinaryAttentionBias(dim=d_model, num_heads=num_heads)

    def forward(self, queries, keys, values, attn_mask, n_vars=None, n_tokens=None, n_pred_vars=None, tau=None, delta=None):
        batch_size, q_len, n_heads, q_head_dim = queries.shape
        _, kv_len, _, kv_head_dim = queries.shape

        scale = self.scale or 1. / sqrt(q_head_dim)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
            
        if self.binary_bias:
            var_id = repeat(torch.arange(n_vars),
                        'C -> (C n_tokens)', n_tokens=n_tokens)
            var_id = repeat(var_id, 'L -> b h L', b=batch_size, h=1).to(queries.device)

            attn_bias = self.attn_bias(var_id, var_id)
            if self.mask_flag:
                if attn_mask is None:
                    if self.covariate:
                        attn_mask = TimerMultivariateWithFutureCovariateMask(batch_size, n_vars, n_tokens, n_pred_vars, device=queries.device)
                    else:
                        attn_mask = TimerMultivariateMask(batch_size, n_vars, n_tokens, device=queries.device)
                attn_mask = attn_bias.masked_fill_(attn_mask.mask, float("-inf"))
            else:
                attn_mask = attn_bias
            scores += attn_mask

        elif self.mask_flag:
            if attn_mask is None:
                if self.covariate:
                    attn_mask = TimerMultivariateWithFutureCovariateMask(batch_size, n_vars, n_tokens, n_pred_vars, device=queries.device)
                else:
                    attn_mask = TimerMultivariateMask(batch_size, n_vars, n_tokens, device=queries.device)
            scores.masked_fill_(attn_mask.mask, float("-inf"))
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
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

        return self.out_projection(out), attn
    

class AttentionLayer_multivariate(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer_multivariate, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, n_vars=None, n_tokens=None, n_pred_vars=None, tau=None, delta=None):
        batch_size, q_len, d_model = queries.shape
        _, kv_len, _ = keys.shape
        n_heads = self.n_heads

        queries = self.query_projection(queries).view(batch_size, q_len, n_heads, -1)
        keys = self.key_projection(keys).view(batch_size, kv_len, n_heads, -1)
        values = self.value_projection(values).view(batch_size, kv_len, n_heads, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            n_vars=n_vars,
            n_tokens=n_tokens,
            n_pred_vars=n_pred_vars,
            tau=tau,
            delta=delta
        )
        out = out.view(batch_size, q_len, -1)

        return self.out_projection(out), attn