#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==========================================================================================
# Main script for training and evaluating a motion prediction model on the CMU Mocap dataset.
#
# This script orchestrates the entire pipeline for a composite model architecture:
#   - EKAE: A primary model for 3D pose sequence prediction.
#   - GADE: An auxiliary model for motion prediction in the frequency domain.
#   - GCN (to_class): A Graph Convolutional Network for action classification.
#
# Core functionalities include:
#   - Loading the CMU Mocap dataset.
#   - Initializing models and the optimizer.
#   - Running training, validation, and a special per-action testing loop.
#   - Providing rich console feedback, logging metrics, and saving the best model.
# ==========================================================================================

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# --- Import 'rich' library for enhanced and beautiful console output ---
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.table import Table
from torch.utils.data import DataLoader

# --- Import project-specific modules ---
from gade.config import config as gade_config
from gade.model import GADE as Model
from model import GCN, EKAE_S1
from utils import cmu_mocap as datasets
from utils import log, util
from utils.opt import Options

# ===============================================================
# GLOBAL CONSTANTS
# Moved from `run_model` to avoid re-creation and improve clarity.
# ===============================================================

# Indices of the dimensions (coordinates) from the CMU dataset to be used by the model.
DIM_USED = np.array(
    [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 42, 43, 44, 45,
     46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 75, 76, 77, 78, 79, 80, 84, 85, 86,
     90, 91, 92, 93, 94, 95, 96, 97, 98, 102, 103, 104, 105, 106, 107, 111, 112, 113]
)

# Anatomical constraints for CMU skeleton: some joints should be copied from others.
# This indexing is adopted from prior work (e.g., MSR-GCN).
JOINT_TO_IGNORE = np.array([16, 20, 29, 24, 27, 33, 36])
INDEX_TO_IGNORE = np.concatenate((JOINT_TO_IGNORE * 3, JOINT_TO_IGNORE * 3 + 1, JOINT_TO_IGNORE * 3 + 2))
JOINT_EQUAL = np.array([15, 15, 15, 23, 23, 32, 32])
INDEX_TO_EQUAL = np.concatenate((JOINT_EQUAL * 3, JOINT_EQUAL * 3 + 1, JOINT_EQUAL * 3 + 2))

# CORE HYPER PARAMETERS AT TRAINING STAGE
ALPHA, BETA, GAMMA = 0.4, 0.4, 1.0


def main(opt):
    """
    The main execution function that sets up and runs the entire experiment.
    """
    console = Console()

    # --- 1. Initialization and Setup ---
    lr_now = opt.lr_now
    start_epoch = 1
    err_best = float('inf')

    # --- 2. Model Creation ---
    console.print("[bold green]>>> Creating models...[/bold green]")
    net_pred = EKAE_S1.EKAE(
        in_features=opt.in_features, kernel_size=opt.kernel_size, d_model=opt.d_model,
        num_stage=opt.num_stage, dct_n=opt.dct_n
    ).to(opt.device)

    gade_config.dim = opt.in_features
    gade_config.motion_mlp.num_layers = 72
    simple_model = Model(gade_config).to(opt.device)
    to_class = GCN.to_class(in_features=opt.in_features * 20, out_features=8).to(opt.device)

    optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, simple_model.parameters()), 'lr': opt.lr_now},
        {'params': filter(lambda p: p.requires_grad, net_pred.parameters()), 'lr': opt.lr_now},
        {'params': filter(lambda p: p.requires_grad, to_class.parameters()), 'lr': opt.lr_now},
    ])
    total_params = sum(p.numel() for p in net_pred.parameters()) / 1e6
    console.print(f"Total params in EKAE model: [bold yellow]{total_params:.2f}M[/bold yellow]")

    # --- 3. Checkpoint Loading ---
    if opt.is_load or opt.is_eval:
        model_path = opt.ckpt
        console.print(f"[bold green]>>> Loading checkpoint from '[cyan]{model_path}[/cyan]'...[/bold green]")
        try:
            ckpt = torch.load(model_path, map_location=opt.device, weights_only=False)
            start_epoch = ckpt['epoch'] + 1
            err_best = ckpt.get('err_best', float('inf'))  # Handle older checkpoints
            lr_now = ckpt['lr']
            net_pred.load_state_dict(ckpt['state_dict'])
            console.print(f"[info]Checkpoint loaded (epoch: {ckpt['epoch']}, best_err: {err_best:.4f})[/info]")
        except FileNotFoundError:
            console.print(
                f"[bold red]Error: Checkpoint file not found at '{model_path}'. Starting from scratch.[/bold red]")

    # --- 4. Pre-computation of DCT Matrices ---
    console.print("[bold green]>>> Pre-computing DCT matrices...[/bold green]")
    dct_m, idct_m = util.get_dct_matrix(opt.input_n)
    dct_matrices = (
        torch.from_numpy(dct_m).float().to(opt.device),
        torch.from_numpy(idct_m).float().to(opt.device)
    )

    # --- 5. Dataset and DataLoader Setup ---
    console.print("[bold green]>>> Loading datasets...[/bold green]")
    num_workers = 4
    if not opt.is_eval:
        dataset = datasets.Datasets(opt, split=0)
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=True)
        valid_dataset = datasets.Datasets(opt, split=1)
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=num_workers,
                                  pin_memory=True)
        console.print(f"Training dataset length: [yellow]{len(dataset)}[/yellow]")
        console.print(f"Validation dataset length: [yellow]{len(valid_dataset)}[/yellow]")

    acts = ["basketball", "basketball_signal", "directing_traffic",
            "jumping", "running", "soccer", "walking", "washwindow"]

    origin_output_n = opt.output_n
    # --- 6. Main Training Loop ---
    if not opt.is_eval:
        for epo in range(start_epoch, opt.epoch + 1):
            console.rule(f"[bold]Epoch {epo}/{opt.epoch}[/bold]", style="blue")
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))

            # -- Train and Validation --
            opt.output_n = origin_output_n  # Set prediction length for train/val
            ret_train = run_model(net_pred, simple_model, to_class, optimizer, data_loader, opt, dct_matrices, epo,
                                  "Training", console)
            ret_valid = run_model(net_pred, simple_model, to_class, None, valid_loader, opt, dct_matrices, epo,
                                  "Validation", console)

            # -- Per-Action Testing --
            opt.output_n = 25
            errs = np.zeros([len(acts), opt.output_n])

            console.print("Start testing per action...")
            for i, act in enumerate(acts):
                console.print(f"[bold blue] >>> Testing action: {act} [/bold blue]")
                test_dataset = datasets.Datasets(opt, split=2, actions=[act])
                test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=True)

                ret_test = run_model(net_pred, simple_model, to_class, None, test_loader, opt, dct_matrices, epo,
                                     f"Testing on [bold]{act}[/bold]", console)
                errs[i] = np.array(list(ret_test.values()))

            errs_mean = np.mean(errs, axis=0)

            # -- Summarize, Log, and Save --
            key_test_indices = [1, 3, 7, 9, 24]
            average_error = np.mean(errs_mean[key_test_indices])

            results_table = Table(title=f"Epoch {epo} Summary (MPJPE in mm)", show_header=True,
                                  header_style="bold magenta")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="yellow")
            results_table.add_row("Train Error", f"{ret_train['m_p3d_h36']:.3f}")
            results_table.add_row("Validation Error", f"{ret_valid['m_p3d_h36']:.3f}")
            results_table.add_row("Test Average Error", f"{average_error:.3f}")
            console.print(results_table)

            # Log to CSV
            log_data_head = ['epoch', 'lr'] + list(ret_train.keys()) + [f'test_{k}' for k in ret_valid.keys()] + [
                f'test_{i * 40}ms' for i in range(1, opt.output_n + 1)]
            log_data_vals = [epo, lr_now] + list(ret_train.values()) + list(ret_valid.values()) + list(errs_mean)
            log.save_csv_log(opt, log_data_head, np.array(log_data_vals), is_create=(epo == 1),
                             file_name="train_pre_action")

            # Save Checkpoint
            is_best = average_error < err_best
            if is_best:
                err_best = average_error
                console.print(f"[bold green]🎉 New best test error: {err_best:.4f} mm. Saving model...[/bold green]")

            test_dict = {f'#{i + 1}': errs_mean[i] for i in range(opt.output_n)}

            log.save_ckpt(
                {'epoch': epo, 'lr': lr_now, 'err': err_best,
                 'state_dict': net_pred.state_dict(), 'optimizer': optimizer.state_dict()},
                is_best=is_best, opt=opt, test_dict=test_dict
            )
    else:
        # --- 7. Evaluation Mode ---
        opt.output_n = 25
        errs = np.zeros([len(acts), opt.output_n])
        console.print("[bold green]>>> Running in evaluation mode...[/bold green]")

        for i, act in enumerate(acts):
            console.print(f"[bold blue] >>> Testing action: {act} [/bold blue]")
            test_dataset = datasets.Datasets(opt, split=2, actions=[act])
            test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)

            ret_test = run_model(net_pred, simple_model, to_class, None, test_loader, opt, dct_matrices, 0,
                                 f"Testing on [bold]{act}[/bold]", console)
            errs[i] = np.array(list(ret_test.values()))

        errs_mean = np.mean(errs, axis=0)

        test_dict = {f'#{i + 1}': errs_mean[i] for i in range(opt.output_n)}

        # Display evaluation results in a formatted table.
        cols_per_row = 5
        rows_per_table = 5
        cells_per_table = cols_per_row * rows_per_table
        items = list(test_dict.items())

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

        # Log results
        log.save_csv_log(opt, list(test_dict.keys()), np.array(list(test_dict.values())),
                         is_create=True, file_name='test_pre_action')
        console.print("[info]Evaluation results saved to CSV.[/info]")


def run_model(
        net_pred: nn.Module,
        simple_model: nn.Module,
        to_class: nn.Module,
        optimizer: Optional[optim.Optimizer],
        data_loader: DataLoader,
        opt,
        dct_matrices: Tuple[torch.Tensor, torch.Tensor],
        epo: int,
        mode: str,
        console: Console
) -> Dict[str, float]:
    """
    Executes a single run of training, validation, or testing for one epoch.
    """
    is_train = optimizer is not None
    is_test = "Test" in mode

    net_pred.train(is_train)
    simple_model.train(is_train)
    to_class.train(is_train)

    total_loss, total_mpjpe, n = 0.0, 0.0, 0

    out_n = opt.output_n if is_test else opt.output_n
    itera = 3 if is_test else 1

    if is_test:
        per_frame_mpjpe = np.zeros(out_n)

    dct_m, idct_m = dct_matrices

    progress = Progress(
        TextColumn(f"[bold cyan]{mode} Epoch {epo}"), BarColumn(bar_width=None),
        MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
        TextColumn("[magenta]{task.fields[info]}"),
        console=console, transient=True
    )

    with progress:
        task = progress.add_task("Processing...", total=len(data_loader), info="")
        for i, (class_gt, p3d_h36) in enumerate(data_loader):
            batch_size = p3d_h36.shape[0]
            if batch_size == 1 and is_train:
                continue

            class_gt = class_gt.to(opt.device)
            p3d_h36 = p3d_h36.float().to(opt.device)
            p3d_src = p3d_h36[:, :, DIM_USED]

            # --- Forward Pass ---
            p3d_out_all = net_pred(p3d_src, input_n=opt.input_n, output_n=10, itera=itera)

            # --- Loss Calculation & Optimization (Training only) ---
            if is_train:
                # GADE + GCN forward pass
                motion_input = torch.matmul(dct_m, p3d_src[:, :opt.input_n, :])
                motion_pred = torch.matmul(idct_m, simple_model(motion_input))[:, -opt.dct_n:, :]
                motion_class_out = to_class(motion_pred.reshape(batch_size, -1))
                class_loss = F.cross_entropy(motion_class_out, class_gt.long())

                # Reshape for loss calculation
                p3d_sup = p3d_src[:, -out_n - opt.kernel_size:].reshape(-1, opt.kernel_size + out_n, len(DIM_USED) // 3,
                                                                        3)
                p3d_out_reshaped = p3d_out_all.reshape(batch_size, opt.kernel_size + out_n, itera, len(DIM_USED) // 3,
                                                       3)
                motion_out_reshaped = motion_pred.reshape(batch_size, opt.kernel_size + 10, -1, 3)

                # Calculate individual losses
                loss_p3d = torch.sum(torch.norm(p3d_out_reshaped[:, :, 0] - p3d_sup, dim=3))
                loss_motion = torch.mean(torch.norm(motion_out_reshaped - p3d_sup, dim=3))

                combined_loss = ALPHA * loss_p3d + BETA * loss_motion + GAMMA * class_loss

                # Backpropagation
                optimizer.zero_grad()
                combined_loss.requires_grad_(True)
                combined_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(simple_model.parameters()),
                    max_norm=opt.max_norm
                )
                optimizer.step()

                total_loss += loss_p3d.item() * batch_size  # Log original p3d loss
                progress.update(task, advance=1, info=f"Loss: {combined_loss.item() / 1000.0:.4f}")
            else:
                progress.update(task, advance=1)

            # --- Metric Calculation (MPJPE) ---
            p3d_out = p3d_h36.clone()[:, opt.input_n: opt.input_n + out_n]
            if is_test:
                p3d_out_all_frames = p3d_out_all[:, opt.kernel_size:].transpose(1, 2).reshape(batch_size, 10 * itera,
                                                                                              -1)[:, :out_n]
                p3d_out[:, :, DIM_USED] = p3d_out_all_frames
            else:
                p3d_out[:, :, DIM_USED] = p3d_out_all[:, opt.kernel_size:, 0]

            p3d_out[:, :, INDEX_TO_IGNORE] = p3d_out[:, :, INDEX_TO_EQUAL]

            p3d_out_reshaped = p3d_out.reshape(-1, out_n, 38, 3)
            p3d_h36_reshaped = p3d_h36.reshape(-1, opt.input_n + out_n, 38, 3)
            ground_truth_frames = p3d_h36_reshaped[:, opt.input_n:]

            if is_test:
                mpjpe_per_frame = torch.sum(
                    torch.mean(torch.norm(ground_truth_frames - p3d_out_reshaped, dim=3), dim=2), dim=0)
                per_frame_mpjpe += mpjpe_per_frame.cpu().data.numpy()
            else:
                total_mpjpe += torch.mean(torch.norm(ground_truth_frames - p3d_out_reshaped, dim=3)).item() * batch_size

            n += batch_size

    # --- Aggregate and Return Results ---
    ret = {}
    if is_train:
        ret["l_p3d"] = total_loss / n

    if is_test:
        per_frame_mpjpe /= n
        for j in range(out_n):
            ret[f"#{j + 1}"] = per_frame_mpjpe[j]
    else:
        ret["m_p3d_h36"] = total_mpjpe / n

    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
