#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import json
import os
import torch
import pandas as pd
import numpy as np


def save_csv_log(opt, head, value, is_create=False, file_name='test'):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    # Judge if ckpt is a directory
    ckpt_dir = opt.ckpt if os.path.isdir(opt.ckpt) else os.path.dirname(opt.ckpt)
    file_path = ckpt_dir + '/{}.csv'.format(file_name)
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, 'a') as f:
            df.to_csv(f, header=False, index=False)


def save_ckpt(state, is_best=True, opt=None, test_dict=None):
    cur_ckpt_name = f"ep{state['epoch']:03d}"
    metrics_to_perceive = ['#2', '#4', '#8', '#10', '#14', '#25']
    for metric in metrics_to_perceive:
        assert metric in test_dict, "Metric {} not found in state['metrics']".format(metric)
        cur_ckpt_name += f"_{test_dict[metric]:.2f}"
    cur_ckpt_name += '.pth.tar'

    file_path = os.path.join(opt.ckpt, cur_ckpt_name)
    torch.save(state, file_path)

    if is_best:
        best_ckpt_name = 'best_' + cur_ckpt_name
        file_path = os.path.join(opt.ckpt, best_ckpt_name)
        torch.save(state, file_path)


def save_options(opt):
    with open(opt.ckpt + '/option.json', 'w') as f:
        f.write(json.dumps(vars(opt), sort_keys=False, indent=4))
