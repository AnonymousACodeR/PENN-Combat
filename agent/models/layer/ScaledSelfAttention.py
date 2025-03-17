import torch
from torch import nn


class VanillaScaledSelfAttention(nn.Module):
    def __init__(self, emb_dim, q_dim=None, v_dim=None):
        super(VanillaScaledSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.q_dim = q_dim if q_dim is not None else self.emb_dim * 10
        self.v_dim = v_dim if v_dim is not None else self.q_dim
        self.w_query = nn.Linear(self.emb_dim, self.q_dim)
        # Remark: q_dim == k_dim by definition
        self.w_key = nn.Linear(self.emb_dim, self.q_dim)
        self.w_value = nn.Linear(self.emb_dim, self.v_dim)
        # self.softmax = nn.Softmax(dim=-2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.w_query(x)
        key = self.w_key(x)
        value = self.w_value(x)
        # return F.scaled_dot_product_attention(query, key, value)
        qk = torch.matmul(query, key.transpose(-2, -1)) / (self.emb_dim ** 0.5)
        # qk = nn.ELU()(qk)
        attention = self.softmax(qk)
        output = torch.matmul(attention, value)
        return output


class ScaledSelfAttention(nn.Module):
    def __init__(self, emb_dim, q_dim=None, v_dim=None, bias=True):
        super(ScaledSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.q_dim = q_dim if q_dim is not None else self.emb_dim * 10
        self.v_dim = v_dim if v_dim is not None else self.q_dim
        self.w_query = nn.Linear(self.emb_dim, self.q_dim, bias=bias)
        # Remark: q_dim == k_dim by definition
        self.w_key = nn.Linear(self.emb_dim, self.q_dim, bias=bias)
        self.w_value = nn.Linear(self.emb_dim, self.v_dim, bias=bias)
        # self.softmax = nn.Softmax(dim=-2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.w_query(x)
        query = nn.Sigmoid()(query)
        key = self.w_key(x)
        value = self.w_value(x)
        # return F.scaled_dot_product_attention(query, key, value)
        qk = torch.matmul(query, key.transpose(-2, -1)) / (self.emb_dim ** 0.5)
        qk = nn.ELU()(qk)
        attention = self.softmax(qk)
        output = torch.matmul(attention, value)
        return output
