#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==========================================================================================
# Main script for training and evaluating a motion prediction model on the 3DPW dataset.
#
# This script orchestrates the entire pipeline for a composite model architecture:
#   - EKAE: A primary model for 3D pose sequence prediction.
#   - GADE: An auxiliary model for motion prediction in the frequency domain.
#   - GCN (to_class): A Graph Convolutional Network for action classification.
#
# Core functionalities include:
#   - Loading the 3DPW dataset.
#   - Initializing the models and optimizer.
#   - Running training, validation, and testing loops with rich console feedback.
#   - Logging metrics and saving the best model checkpoints.
# ==========================================================================================

import math
import time
from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# --- Import 'rich' library for enhanced and beautiful console output ---
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.table import Table
from torch.utils.data import DataLoader

# --- Import project-specific modules ---
from gade.config import config as gade_config
from gade.model import GADE as Model
from model import GCN, EKAE
from utils import dpw3d as datasets
from utils import log, util
from utils.opt import Options

# CORE HYPER PARAMETERS AT TRAINING STAGE
ALPHA, BETA, GAMMA = 0.5, 0.5, 1.0


def main(opt):
    """
    The main execution function that sets up and runs the entire experiment.
    """
    # Initialize the rich console for beautiful printing.
    console = Console()

    # --- 1. Initialization and Setup ---
    lr_now = opt.lr_now
    start_epoch = 1
    err_best = float('inf')

    # --- 2. Model Creation ---
    console.print("[bold green]>>> Creating models...[/bold green]")

    # Initialize the primary prediction model (EKAE).
    net_pred = EKAE.EKAE(
        in_features=opt.in_features,
        kernel_size=opt.kernel_size,
        d_model=opt.d_model,
        num_stage=opt.num_stage,
        dct_n=opt.dct_n
    ).to(opt.device)

    # Initialize the auxiliary motion prediction model (GADE).
    gade_model = Model(gade_config).to(opt.device)

    # Initialize the action classification model (GCN).
    to_class = GCN.to_class(in_features=66 * 20, out_features=30).to(opt.device)

    # Calculate and display the number of trainable parameters.
    total_params = sum(p.numel() for p in net_pred.parameters()) / 1e6
    console.print(f"Total params in EKAE model: [bold yellow]{total_params:.2f}M[/bold yellow]")

    # Set up the Adam optimizer to train all three models jointly.
    optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, gade_model.parameters()), 'lr': opt.lr_now},
        {'params': filter(lambda p: p.requires_grad, net_pred.parameters()), 'lr': opt.lr_now},
        {'params': filter(lambda p: p.requires_grad, to_class.parameters()), 'lr': opt.lr_now},
    ])

    # --- 3. Checkpoint Loading (if specified) ---
    if opt.is_load or opt.is_eval:
        model_path = opt.ckpt
        console.print(f"[bold green]>>> Loading checkpoint from '[cyan]{model_path}[/cyan]'...[/bold green]")
        try:

            ckpt = torch.load(model_path, map_location=opt.device, weights_only=False)
            # Drop state_dict: "one_mlp.linear.weight", "one_mlp.linear.bias", "one_mlp2.linear.weight", "one_mlp2.linear.bias"

            for key in list(ckpt['state_dict'].keys()):
                if key.startswith('one_mlp.linear') or key.startswith('one_mlp2.linear'):
                    del ckpt['state_dict'][key]

            start_epoch = ckpt['epoch'] + 1
            err_best = ckpt['err']
            lr_now = ckpt['lr']
            net_pred.load_state_dict(ckpt['state_dict'])
            console.print(f"[info]Checkpoint loaded (epoch: {ckpt['epoch']}, best_err: {err_best:.4f})[/info]")
        except FileNotFoundError:
            console.print(
                f"[bold red]Error: Checkpoint file not found at '{model_path}'. Starting from scratch.[/bold red]")

    # --- 4. Pre-computation of DCT Matrices (Performance Optimization) ---
    # This avoids redundant calculations inside the training loop.
    console.print("[bold green]>>> Pre-computing DCT matrices...[/bold green]")
    dct_m, idct_m = util.get_dct_matrix(opt.input_n)
    dct_matrices = (
        torch.from_numpy(dct_m).float().to(opt.device),
        torch.from_numpy(idct_m).float().to(opt.device)
    )

    # --- 5. Dataset and DataLoader Setup ---
    console.print("[bold green]>>> Loading datasets...[/bold green]")
    # For better performance, set num_workers > 0. 4 is a common default.
    # Pinned memory helps speed up CPU to GPU data transfers.
    num_workers = 4

    if not opt.is_eval:
        dataset = datasets.Datasets(opt, split=0)
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=True)
        console.print(f"Training dataset length: [yellow]{len(dataset)}[/yellow]")

        valid_dataset = datasets.Datasets(opt, split=1)
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=num_workers,
                                  pin_memory=True)
        console.print(f"Validation dataset length: [yellow]{len(valid_dataset)}[/yellow]")

    opt_for_test = deepcopy(opt)
    opt_for_test.output_n = 25
    test_dataset = datasets.Datasets(opt_for_test, split=2)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)
    console.print(f"Testing dataset length: [yellow]{len(test_dataset)}[/yellow]")

    # --- 6. Evaluation-Only Mode ---
    if opt.is_eval:

        opt.output_n = 25
        ret_test = run_model(
            net_pred,
            gade_model,
            to_class,
            data_loader=test_loader,
            opt=opt,
            dct_matrices=dct_matrices,
            mode="Testing",
            console=console,
        )

        # Display evaluation results in a formatted table.
        cols_per_row = 5
        rows_per_table = 5
        cells_per_table = cols_per_row * rows_per_table

        items = list(ret_test.items())

        for idx in range(0, len(items), cells_per_table):
            chunk = items[idx:idx + cells_per_table]

            table = Table(
                title=f"[bold]Evaluation Results (MPJPE) Part {idx // cells_per_table + 1}[/bold]",
                style="cyan"
            )

            # 添加列
            for i in range(cols_per_row):
                table.add_column(f"Frame", justify="center")
                table.add_column("Error", justify="center")

            # 按行填充
            for r in range(0, len(chunk), cols_per_row):
                row_cells = []
                for k, v in chunk[r:r + cols_per_row]:
                    title = int(k.replace('#', '')) * 40
                    row_cells.append(f'{title} ms')
                    row_cells.append(f"[bold yellow]{v:.2f}[/bold yellow]")
                # 不足的列补空
                while len(row_cells) < cols_per_row * 2:
                    row_cells.extend(["", ""])
                table.add_row(*row_cells)

            console.print(table)

        # Save results to CSV.
        log.save_csv_log(opt, list(ret_test.keys()), np.array(list(ret_test.values())),
                         is_create=True, file_name='test_pre_action')
        console.print("[info]Evaluation results saved to CSV.[/info]")
        return

    # --- 7. Training Mode ---
    console.print("[bold blue]>>> Starting Training...[/bold blue]")

    original_output_n = opt.output_n
    for epo in range(start_epoch, opt.epoch + 1):
        epoch_start_time = time.time()
        console.rule(f"[bold]Epoch {epo}/{opt.epoch}[/bold]", style="blue")

        lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
        opt.output_n = original_output_n
        # Run one epoch of training.
        ret_train = run_model(net_pred, gade_model, to_class, data_loader=data_loader, opt=opt,
                              dct_matrices=dct_matrices, optimizer=optimizer, epo=epo, console=console)

        # Run validation.
        ret_valid = run_model(net_pred, gade_model, to_class, data_loader=valid_loader, opt=opt,
                              dct_matrices=dct_matrices, epo=epo, mode="Validation", console=console)

        opt.output_n = 25
        # Run testing.
        ret_test = run_model(net_pred, gade_model, to_class, data_loader=test_loader, opt=opt,
                             dct_matrices=dct_matrices, epo=epo, mode="Testing", console=console)

        epoch_time = time.time() - epoch_start_time

        # Summarize and display epoch results in a table.
        results_table = Table(title=f"Epoch {epo} Summary", style="green")
        results_table.add_column("Metric", style="cyan", no_wrap=True)
        results_table.add_column("Train", style="magenta")
        results_table.add_column("Validation", style="yellow")
        results_table.add_column("Test (80ms)", style="red")

        results_table.add_row("MPJPE (mm)", f"{ret_train['m_p3d_h36']:.2f}", f"{ret_valid['m_p3d_h36']:.2f}",
                              f"{ret_test['#2']:.2f}")
        results_table.add_row("Combined Loss", f"{ret_train['l_p3d']:.4f}", "-", "-")
        console.print(results_table)
        console.print(
            f"Epoch completed in [bold_purple]{epoch_time:.2f}s[/bold_purple]. Current LR: [bold_purple]{lr_now:.8f}[/bold_purple]")

        # Prepare and save log data.
        log_data = {'epoch': epo, 'lr': lr_now, **ret_train}
        log_data.update({f'valid_{k}': v for k, v in ret_valid.items()})
        log_data.update({f'test_{k}': v for k, v in ret_test.items()})
        log.save_csv_log(
            opt,
            list(log_data.keys()),
            np.array(list(log_data.values())),
            is_create=(epo == 1),
            file_name='train_pre_action'
        )

        # Check for the best model and save checkpoint.
        is_best = ret_valid['m_p3d_h36'] < err_best
        if is_best:
            err_best = ret_valid['m_p3d_h36']
            console.print(f"[bold green]🎉 New best validation error: {err_best:.2f} mm. Saving model...[/bold green]")

        log.save_ckpt({
            'epoch': epo, 'lr': lr_now, 'err': err_best,
            'state_dict': net_pred.state_dict(), 'optimizer': optimizer.state_dict()
        }, is_best=is_best, opt=opt, test_dict=ret_test)


def run_model(
        net_pred: nn.Module,
        gade_model: nn.Module,
        to_class: nn.Module,
        data_loader: DataLoader,
        opt,
        dct_matrices: Tuple[torch.Tensor, torch.Tensor],
        console: Console,
        optimizer: Optional[optim.Optimizer] = None,
        epo: int = 1,
        mode: str = "Training"
) -> Dict[str, float]:
    """
    Executes a single epoch of training, validation, or testing.

    Args:
        net_pred: The primary pose prediction model (EKAE).
        gade_model: The auxiliary motion prediction model (GADE).
        to_class: The action classification model (GCN).
        data_loader: The DataLoader for the current dataset split.
        opt: Configuration options.
        dct_matrices: A tuple of pre-computed DCT and IDCT matrices.
        console: The rich console instance for progress bar display.
        optimizer: The optimizer. Required for training.
        epo: The current epoch number (for display purposes).
        mode: A string indicating the current mode ("Training", "Validation", etc.).

    Returns:
        A dictionary containing the calculated metrics for the epoch.
    """
    # Determine if this is a training run.
    is_train = (optimizer is not None)
    # Determine if this is a test run (which has different evaluation logic).
    is_test = "Test" in mode

    # Set models to the correct mode.
    net_pred.train(is_train)
    gade_model.train(is_train)
    to_class.train(is_train)

    # --- Initialize accumulators for metrics ---
    total_loss = 0.
    total_mpjpe = 0.
    total_samples = 0
    if is_test:
        per_frame_mpjpe = np.zeros(opt.output_n)

    # Unpack DCT matrices.
    dct_m, idct_m = dct_matrices

    # Configure the rich progress bar.
    progress = Progress(
        TextColumn(f"[bold cyan]{mode} Epoch {epo}"), BarColumn(bar_width=None),
        MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
        TextColumn("[magenta]{task.fields[info]}"),
        console=console, transient=True
    )

    with progress:
        task = progress.add_task("Processing...", total=len(data_loader), info="")

        for i, (p3d_h36_raw, class_gt) in enumerate(data_loader):
            # --- 1. Data Preparation ---
            batch_size = p3d_h36_raw.shape[0]
            if batch_size == 1 and is_train:
                continue

            # Reshape and move data to the target device, scale by 1000.
            p3d_h36 = p3d_h36_raw.flatten(2).float().to(opt.device) * 1000
            p3d_src = p3d_h36.clone()  # Will be used as input for models.

            # --- 2. Forward Pass ---
            # The model predicts 10 frames at a time iteratively.
            itera = math.ceil(opt.output_n / 10)

            p3d_out_all_raw = net_pred(p3d_src, input_n=opt.input_n, output_n=10, itera=itera)

            # --- 3. Loss Calculation and Optimization (Training only) ---
            if is_train:
                # Forward pass for auxiliary models (GADE and GCN).
                motion_input_dct = torch.matmul(dct_m, p3d_src[:, :opt.input_n, :])
                motion_pred_dct = gade_model(motion_input_dct)
                motion_pred = torch.matmul(idct_m, motion_pred_dct)[:, -20:, :]

                motion_class_in = motion_pred.reshape(batch_size, -1)
                motion_class_out = to_class(motion_class_in)
                class_gt = class_gt.to(opt.device).long()

                # Reshape ground truth and predictions for loss calculation.
                p3d_sup = p3d_h36[:, -opt.output_n - opt.kernel_size:].reshape(-1, opt.kernel_size + opt.output_n,
                                                                               opt.in_features // 3, 3)
                p3d_out_all_train = p3d_out_all_raw.reshape(batch_size, opt.kernel_size + opt.output_n,
                                                            opt.in_features // 3, 3)
                motion_out_all = motion_pred.reshape(batch_size, opt.kernel_size + 10, -1,
                                                     3)  # Note: GADE output length is fixed at 10.

                # Calculate the three-part loss.
                loss_p3d = F.l1_loss(p3d_out_all_train, p3d_sup)
                loss_motion = F.l1_loss(motion_out_all,
                                        p3d_sup[:, :, :opt.in_features // 3, :])  # Match shapes for loss
                loss_class = F.cross_entropy(motion_class_out, class_gt)

                # Combine losses with weights.
                combined_loss = ALPHA * loss_p3d + BETA * loss_motion + GAMMA * loss_class

                # Backpropagation.
                optimizer.zero_grad()
                combined_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(gade_model.parameters()) + list(net_pred.parameters()) + list(to_class.parameters()),
                    max_norm=opt.max_norm)
                optimizer.step()

                total_loss += combined_loss.item() * batch_size
                progress.update(task, advance=1, info=f"Loss: {combined_loss.item():.4f}")
            else:
                progress.update(task, advance=1)

            # --- 4. Metric Calculation (MPJPE) ---
            # Reconstruct full 3D pose from model output.
            p3d_out = p3d_h36.clone()[:, opt.input_n:]
            if is_test:
                # Special reshaping for the test protocol.
                p3d_out_all = p3d_out_all_raw[:, opt.kernel_size:, :].transpose(1, 2)
                p3d_out_all = p3d_out_all.reshape(batch_size, itera * 10, -1)[:, :opt.output_n]
            else:
                p3d_out_all = p3d_out_all_raw[:, opt.kernel_size:, 0]

            p3d_out[:, :, :opt.in_features] = p3d_out_all

            # Reshape for metric calculation (B, T, J, C).
            p3d_out_reshaped = p3d_out.reshape(batch_size, opt.output_n, opt.in_features // 3, 3)
            p3d_h36_reshaped = p3d_h36.reshape(batch_size, opt.input_n + opt.output_n, opt.in_features // 3, 3)

            ground_truth_frames = p3d_h36_reshaped[:, opt.input_n:]

            if is_test:
                # Per-frame MPJPE for testing.
                mpjpe_per_frame = torch.mean(torch.norm(ground_truth_frames - p3d_out_reshaped, dim=3), dim=2).sum(
                    dim=0)
                per_frame_mpjpe += mpjpe_per_frame.cpu().data.numpy()
            else:
                # Overall MPJPE for training/validation.
                mpjpe = torch.mean(torch.norm(ground_truth_frames - p3d_out_reshaped, dim=3))
                total_mpjpe += mpjpe.item() * batch_size

            total_samples += batch_size

    # --- 5. Aggregate and Return Results ---
    ret = {}
    if is_train:
        ret["l_p3d"] = total_loss / total_samples

    if is_test:
        per_frame_mpjpe /= total_samples
        for j in range(opt.output_n):
            ret[f"#{j + 1}"] = per_frame_mpjpe[j]
    else:
        ret["m_p3d_h36"] = total_mpjpe / total_samples

    return ret


if __name__ == '__main__':
    # Entry point of the script. Parses command-line arguments and calls main.
    option = Options().parse()
    main(option)
