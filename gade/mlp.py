import math

import torch
from torch import nn
import torch.nn.functional as F


def make_quaternion_mul(kernel):
    """" The constructed 'hamilton' W is a modified version of the quaternion representation,
        thus doing tf.matmul(Input,W) is equivalent to W * Inputs. """
    dim = kernel.size(1) // 4
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=1)
    assert kernel.size(1) == hamilton.size(1)
    return hamilton


class GAMLP(nn.Module):
    def __init__(self, in_features, out_features, dropout, quaternion_ff=True, act=True):
        super(GAMLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quaternion_ff = quaternion_ff
        self.act = act
        self.dropout = nn.Dropout(dropout)

        if self.quaternion_ff:
            self.weight = nn.Parameter(torch.FloatTensor(self.in_features // 4, self.out_features))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, double_type_used_in_graph=False):
        x = self.dropout(input).float()  # Current Pytorch 1.5.0 doesn't support Dropout for sparse matrix

        if self.quaternion_ff:
            hamilton = make_quaternion_mul(self.weight)
            if double_type_used_in_graph:  # to deal with scalar type between node and graph classification tasks
                hamilton = hamilton.double()
                x = x.double()
            support = torch.mm(x, hamilton)  # Hamilton product, quaternion multiplication!
        else:
            support = torch.mm(x, self.weight)

        if double_type_used_in_graph:  # to deal with scalar type between node and graph classification tasks
            support = support.double()

        if self.act:
            support = F.relu(support)
        return support


class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 100]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 100]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=0, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=0, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class Spatial_FC(nn.Module):
    def __init__(self, dim):
        super(Spatial_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)
        # self.arr0 = Rearrange('b n d -> b d n')
        self.arr0 = self.arr0.transpose(1, 2)
        # self.arr1 = Rearrange('b d n -> b n d')
        self.arr1 = self.arr1.transpose(1, 2)

    def forward(self, x):
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
        return x


class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = GAMLP(100, 100, 0.3)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLPblock(nn.Module):

    def __init__(self, dim, seq, use_norm=False, use_spatial_fc=False, layernorm_axis='spatial'):
        super().__init__()

        self.fc0 = Temporal_FC(seq)  # Spatial
        if use_norm:
            if layernorm_axis == 'spatial':
                self.norm0 = LN(dim)
            elif layernorm_axis == 'temporal':
                self.norm0 = LN(dim)
            elif layernorm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)

    def forward(self, x):

        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_

        return x


class TransMLP(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis):
        super().__init__()
        self.mlps = nn.Sequential(*[
            MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis)
            for i in range(num_layers)])

    def forward(self, x):
        x = self.mlps(x)
        return x


def build_mlps(args):
    if 'seq_len' in args:
        seq_len = args.seq_len
    else:
        seq_len = None
    return TransMLP(
        dim=args.hidden_dim,
        seq=seq_len,
        use_norm=args.with_normalization,
        use_spatial_fc=args.spatial_fc_only,
        num_layers=args.num_layers,
        layernorm_axis=args.norm_axis,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    if activation == 'silu':
        return nn.SiLU
    # if activation == 'swish':
    #    return nn.Hardswish
    if activation == 'softplus':
        return nn.Softplus
    if activation == 'tanh':
        return nn.Tanh
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_norm_fn(norm):
    if norm == "batchnorm":
        return nn.BatchNorm1d
    if norm == "layernorm":
        return nn.LayerNorm
    if norm == 'instancenorm':
        return nn.InstanceNorm1d
    raise RuntimeError(F"norm should be batchnorm/layernorm, not {norm}.")
