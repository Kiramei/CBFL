#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime
import os
import sys
import torch
import argparse

from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from utils import log


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--is_eval', dest='is_eval', action='store_true',
                                 help='whether it is to evaluate the model')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')
        self.parser.add_argument('--skip_rate', type=int, default=5, help='skip rate of samples')
        self.parser.add_argument('--skip_rate_test', type=int, default=5, help='skip rate of samples for test')
        self.parser.add_argument('--data_dir_cmu', type=str, default='datasets/CMU')
        self.parser.add_argument('--data_dir_3dpw', type=str, default='datasets/3dpw',
                                 help='path to 3DPW dataset')
        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--in_features', type=int, default=66, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=15, help='size of each model layer')
        self.parser.add_argument('--d_model', type=int, default=256, help='past frame number')
        self.parser.add_argument('--kernel_size', type=int, default=10, help='past frame number')

        self.parser.add_argument('--J', type=int, default=1, help='The number of wavelet filters')
        self.parser.add_argument('--tree_num', type=int, default=2, help='The number of scattering tree')
        self.parser.add_argument('--edge_prob', type=float, default=0.4, help='The probability of edge preservation')
        self.parser.add_argument('--W_pg', type=float, default=0.6,
                                 help='The weight of information propagation between part')
        self.parser.add_argument('--W_p', type=float, default=0.6, help='The weight of part on the whole body')
        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--input_n', type=int, default=50, help='past frame number')
        self.parser.add_argument('--output_n', type=int, default=10, help='future frame number')
        self.parser.add_argument('--dct_n', type=int, default=20, help='future frame number')
        self.parser.add_argument('--lr_now', type=float, default=0.005)
        self.parser.add_argument('--max_norm', type=float, default=10000)
        self.parser.add_argument('--epoch', type=int, default=50)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--test_batch_size', type=int, default=32)
        self.parser.add_argument('--is_load', dest='is_load', action='store_true',
                                 help='whether to load existing model')

        self.parser.add_argument('--actions', type=str, default='all', help='path to save checkpoint')
        self.parser.add_argument('--train_batch', type=int, default=32)
        self.parser.add_argument('--test_batch', type=int, default=128)
        self.parser.add_argument('--job', type=int, default=10, help='subprocesses to use for data loading')
        self.parser.add_argument('--sample_rate', type=int, default=2, help='frame sampling rate')
        self.parser.add_argument('--is_norm_dct', dest='is_norm_dct', action='store_true',
                                 help='whether to normalize the dct coeff')
        self.parser.add_argument('--is_norm', dest='is_norm', action='store_true',
                                 help='whether to normalize the angles/3d coordinates')
        self.parser.add_argument('--dropout', type=float, default=0.5,
                                 help='dropout probability, 1.0 to make no dropout')
        self.parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', )

        self.parser.set_defaults(max_norm=True)
        self.parser.set_defaults(is_load=False)

    def _print(self):
        console = Console()

        config_table = Table(title="[bold]Configuration Overview[/bold]", show_header=False, box=None)
        config_table.add_column("Parameter", style="cyan", no_wrap=True)
        config_table.add_column("Value", style="")

        def highlight(val):
            if isinstance(val, bool):
                return f"[green]{val}[/green]" if val else f"[red]{val}[/red]"
            elif isinstance(val, (int, float)):
                return f"[cyan]{val}[/cyan]"
            elif isinstance(val, str):
                if "/" in val or "\\" in val:
                    return f"[blue]{val}[/blue]"
                return f"[yellow]{val}[/yellow]"
            else:
                return str(val)

        for key, value in vars(self.opt).items():
            config_table.add_row(str(key), highlight(value))

        console.print(Panel(config_table, border_style="green"))

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()

        if not self.opt.is_eval:
            script_name = os.path.basename(sys.argv[0])[:-3].upper()
            time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            log_name = 'RUN_{}_{}'.format(script_name, time_str)
            self.opt.exp = log_name
            # do some pre-check
            ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
                log.save_options(self.opt)
            self.opt.ckpt = ckpt
            log.save_options(self.opt)
        self._print()
        return self.opt
