#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.autograd.variable import Variable

from utils import forward_kinematics


def readCSVasFloat(filename):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray


"""all methods above are borrowed from https://github.com/una-dinosauria/human-motion-prediction"""


def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = Variable(torch.eye(3, 3).repeat(n, 1, 1)).float().cuda() + torch.mul(
        torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
        (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    return R


def expmap2xyz_torch(expmap):
    """
    convert expmaps to joint locations
    :param expmap: N*99
    :return: N*32*3
    """
    parent, offset, rotInd, expmapInd = forward_kinematics._some_variables()
    xyz = forward_kinematics.fkl_torch(expmap, parent, offset, rotInd, expmapInd)
    return xyz


def find_indices_256(frame_num1, frame_num2, seq_len, input_n=10):
    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = frame_num1 - 150
    T2 = frame_num2 - 150  # seq_len
    idxo1 = None
    idxo2 = None
    for _ in np.arange(0, 128):
        idx_ran1 = rng.randint(16, T1)
        idx_ran2 = rng.randint(16, T2)
        idxs1 = np.arange(idx_ran1 + 50 - input_n, idx_ran1 + 50 - input_n + seq_len)
        idxs2 = np.arange(idx_ran2 + 50 - input_n, idx_ran2 + 50 - input_n + seq_len)
        if idxo1 is None:
            idxo1 = idxs1
            idxo2 = idxs2
        else:
            idxo1 = np.vstack((idxo1, idxs1))
            idxo2 = np.vstack((idxo2, idxs2))
    return idxo1, idxo2

def expmap2xyz_torch_cmu(expmap):
    parent, offset, rotInd, expmapInd = forward_kinematics._some_variables_cmu()
    xyz = forward_kinematics.fkl_torch(expmap, parent, offset, rotInd, expmapInd)
    return xyz
