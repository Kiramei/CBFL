import copy

from einops.layers.torch import Rearrange
from torch import nn

from gade.mlp import build_mlps


class GADE(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(GADE, self).__init__()
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.joints = self.config.dim
        self.q4_dim = self.config.q4_dim
        self.out_len = self.config.out_len
        self.trans1 = nn.Linear(self.joints, self.q4_dim)
        self.trans2 = nn.Linear(self.q4_dim, self.joints)

        self.motion_mlp = build_mlps(self.config.motion_mlp)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.trans2.weight, gain=1e-8)
        nn.init.constant_(self.trans2.bias, 0)

    def forward(self, motion_input):
        motion_feats = self.trans1(motion_input)
        motion_feats = self.motion_mlp(motion_feats.reshape(-1, self.q4_dim))
        motion_feats = self.trans2(motion_feats).reshape(-1, self.out_len, self.joints)



        return motion_feats


class Aggreagation(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(GADE, self).__init__()

        self.trans1 = nn.Linear(self.joints, self.config.q4_dim)
        self.trans2 = nn.Linear(self.config.q4_dim, self.joints)


        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.trans2.weight, gain=1e-8)
        nn.init.constant_(self.trans2.bias, 0)

    def forward(self, motion_input,simlp):
        out1=simlp(motion_input)
        motion_feats = self.trans1(out1)


        return motion_feats

