from collections import OrderedDict
from turtle import forward

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CrossModalContextTransformerEncoderLayer(nn.Module):
    def __init__(self, opt,  d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()

        self.with_pos_embed = PositionalEncoder(opt, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_key_padding_mask, need_weights=False):
        q = k = self.with_pos_embed(src)
        # Post-LN Transformer
        src2, attn_weights = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class PositionalEncoder(nn.Module):
    def __init__(self, opt, d_model):
        super().__init__()
        hidden_dim = d_model
        num_total = 46
        self.src_pos_embed = nn.Embedding(num_total, hidden_dim)
 
    def forward(self, src):
        bsize = src.size(0)
        src_pos = self.src_pos_embed.weight.unsqueeze(0).repeat(bsize, 1, 1)
        src = src + src_pos
        return src