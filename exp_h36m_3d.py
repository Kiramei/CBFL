#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==========================================================================================
# Main script for training and evaluating a motion prediction model on the Human3.6M dataset.
#
# This script orchestrates the entire pipeline for a composite model architecture:
#   - EKAE: A primary model for 3D pose sequence prediction.
#   - GADE: An auxiliary model for motion prediction in the frequency domain.
#   - GCN (to_class): A Graph Convolutional Network for action classification.
#
# Core functionalities include:
#   - Loading the Human3.6M dataset.
#   - Initializing the models and optimizer.
#   - Running training, validation, and testing loops with rich console feedback.
#   - Logging metrics and saving the best model checkpoints.
# ==========================================================================================

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
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from torch.utils.data import DataLoader

# --- Import local modules from the project structure ---
from model import EKAE
from model import GCN
from gade.config import config
from gade.model import GADE as Model
from utils import h36motion3d as datasets
from utils import log
from utils import util
from utils.opt import Options

# ===============================================================
# GLOBAL CONSTANTS
# These are defined globally to avoid re-creation inside loops, improving efficiency.
# ===============================================================

# Indices of the dimensions (coordinates) from the H3.6M dataset to be used by the model.
# The H3.6M dataset contains 96 dimensions (32 joints * 3 coordinates), but many are
# static or redundant. This array selects the relevant, dynamic dimensions.
DIM_USED = np.array(
    [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
     26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
     46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
     75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92]
)

# In the H3.6M dataset, some joints have identical 3D coordinates to others.
# This section defines a post-processing rule to enforce this anatomical constraint.

# Indices of joints whose coordinates should be ignored and copied from other joints.
JOINTS_TO_IGNORE = np.array([16, 20, 23, 24, 28, 31])
# Corresponding flattened dimension indices (x, y, z) for the joints to be ignored.
INDEX_TO_IGNORE = np.concatenate(
    (JOINTS_TO_IGNORE * 3, JOINTS_TO_IGNORE * 3 + 1, JOINTS_TO_IGNORE * 3 + 2)
)
# Indices of joints from which the coordinates will be copied.
JOINTS_EQUAL = np.array([13, 19, 22, 13, 27, 30])
# Corresponding flattened dimension indices for the source joints.
INDEX_TO_EQUAL = np.concatenate(
    (JOINTS_EQUAL * 3, JOINTS_EQUAL * 3 + 1, JOINTS_EQUAL * 3 + 2)
)

# CORE HYPER PARAMETERS AT TRAINING STAGE
ALPHA, BETA, GAMMA = 0.7, 0.3, 0.5


# ===============================================================

def main(opt):
    """
    The main execution function that sets up and runs the entire experiment.
    """
    # Initialize the rich console for beautiful printing.
    console = Console()

    # --- 1. Initialization and Setup ---
    lr_now = opt.lr_now  # Set the initial learning rate.
    start_epoch = 1  # Start from the first epoch unless a checkpoint is loaded.
    err_best = float('inf')  # Initialize the best validation error to infinity.

    # --- 2. Model Creation ---
    console.print("[bold green]>>> Creating models...[/bold green]")

    # Initialize the main prediction model (EKAE).
    net_pred = EKAE.EKAE(
        in_features=opt.in_features,
        kernel_size=opt.kernel_size,
        d_model=opt.d_model,
        num_stage=opt.num_stage,
        dct_n=opt.dct_n,
    ).to(opt.device)

    # Initialize the auxiliary motion prediction model (GADE).
    gade_model = Model(config).to(opt.device)
    # Initialize the action classification model (GCN).
    to_class = GCN.to_class(in_features=66 * 20, out_features=15).to(opt.device)

    # Calculate and display the number of trainable parameters in the main model.
    total_params = sum(p.numel() for p in net_pred.parameters()) / 1e6
    console.print(f"EKAE total params: [bold yellow]{total_params:.2f}M[/bold yellow]")

    # Set up the Adam optimizer to train all three models jointly.
    # It takes parameters from all models that require gradients.
    optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, gade_model.parameters()), 'lr': opt.lr_now},
        {'params': filter(lambda p: p.requires_grad, net_pred.parameters()), 'lr': opt.lr_now},
        {'params': filter(lambda p: p.requires_grad, to_class.parameters()), 'lr': opt.lr_now},
    ])

    # --- 3. Checkpoint Loading (if specified) ---
    if opt.is_load or opt.is_eval:
        model_path_len = opt.ckpt
        console.print(f"[bold green]>>> Loading checkpoint from '[cyan]{model_path_len}[/cyan]'...[/bold green]")
        try:
            # Load the checkpoint from the specified path.
            ckpt = torch.load(model_path_len, map_location=opt.device)
            # Restore the state from the checkpoint.
            start_epoch = ckpt['epoch'] + 1  # Start from the next epoch.
            err_best = ckpt['err']  # Restore the best recorded error.
            lr_now = ckpt['lr']  # Restore the learning rate.
            net_pred.load_state_dict(ckpt['state_dict'])  # Load model weights.
            console.print(f"[info]Checkpoint loaded (epoch: {ckpt['epoch']}, best_err: {err_best:.4f})[/info]")
        except FileNotFoundError:
            # Handle cases where the checkpoint file does not exist.
            console.print(
                f"[bold red]Error: Checkpoint file not found at '{model_path_len}'. Starting from scratch.[/bold red]")

    # --- 4. Pre-computation of DCT Matrices (Performance Optimization) ---
    # The Discrete Cosine Transform (DCT) and its inverse (IDCT) are used by GADE.
    # Pre-computing these matrices outside the training loop avoids redundant calculations.
    dct_m, idct_m = util.get_dct_matrix(opt.input_n)
    dct_matrices = (
        torch.from_numpy(dct_m).float().to(opt.device),
        torch.from_numpy(idct_m).float().to(opt.device)
    )

    # --- 5. Dataset and DataLoader Setup ---
    console.print("[bold green]>>> Loading datasets...[/bold green]")
    if not opt.is_eval:
        # If not in evaluation-only mode, load training and validation data.
        dataset = datasets.Datasets(opt, split=0)  # split=0 for training
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        console.print(f"Training dataset length: [yellow]{len(dataset)}[/yellow]")

        valid_dataset = datasets.Datasets(opt, split=1)  # split=1 for validation
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=4,
                                  pin_memory=True)
        console.print(f"Validation dataset length: [yellow]{len(valid_dataset)}[/yellow]")

    # Load the test dataset. A deepcopy of options is used to set a different
    # prediction length (25 frames) specifically for the standard test protocol.
    opt_for_test = deepcopy(opt)
    opt_for_test.output_n = 25
    test_dataset = datasets.Datasets(opt_for_test, split=2)  # split=2 for testing
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=4,
                             pin_memory=True)
    console.print(f"Testing dataset length: [yellow]{len(test_dataset)}[/yellow]")

    # --- 6. Evaluation-Only Mode ---
    if opt.is_eval:
        # Run the model on the test set.
        ret_test = run_model(
            net_pred, gade_model, to_class,
            data_loader=test_loader,
            opt=opt,
            dct_matrices=dct_matrices,
            is_train=False,
            console=console,
            is_test=True,
            mode='Testing'
        )

        cols_per_row = 5
        rows_per_table = 5
        cells_per_table = cols_per_row * rows_per_table

        items = list(ret_test.items())

        for idx in range(0, len(items), cells_per_table):
            chunk = items[idx:idx + cells_per_table]

            table = Table(
                title=f"[bold]Overview Evaluation Results (MPJPE) Part {idx // cells_per_table + 1}[/bold]",
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

        head = np.array(['act'])
        for k in range(1, opt_for_test.output_n + 1):
            head = np.append(head, [f'#{k}'])

        acts = ["walking", "eating", "smoking", "discussion", "directions",
                "greeting", "phoning", "posing", "purchases", "sitting",
                "sittingdown", "takingphoto", "waiting", "walkingdog",
                "walkingtogether"]
        errs = np.zeros([len(acts) + 1, opt_for_test.output_n])
        for i, act in enumerate(acts):
            test_dataset = datasets.Datasets(opt_for_test, split=2, actions=[act])

            test_loader = DataLoader(
                test_dataset,
                batch_size=opt.test_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

            ret_test = run_model(
                net_pred, gade_model, to_class,
                data_loader=test_loader,
                opt=opt,
                dct_matrices=dct_matrices,
                is_train=False,
                console=console,
                is_test=True,
                mode=f'Testing on {act}'
            )

            ret_log = np.array([])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
            errs[i] = ret_log
        errs[-1] = np.mean(errs[:-1], axis=0)
        acts = np.expand_dims(np.array(acts + ["average"]), axis=1)
        value = np.concatenate([acts, errs.astype(np.str_)], axis=1)
        log.save_csv_log(opt, head, value, is_create=True, file_name='test_pre_action')

        console.print("[info]Evaluation results saved to CSV.[/info]")
        return  # Exit after evaluation.

    # --- 7. Training Mode ---
    console.print("[bold blue]>>> Starting Training...[/bold blue]")
    for epo in range(start_epoch, opt.epoch + 1):
        # Display a rule line to separate epochs in the console.
        console.rule(f"[bold]Epoch {epo}/{opt.epoch}[/bold]", style="blue")

        # Apply learning rate decay for the current epoch.
        lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))

        # Run one epoch of training.
        ret_train = run_model(
            net_pred, gade_model, to_class,
            data_loader=data_loader,
            opt=opt,
            dct_matrices=dct_matrices,
            optimizer=optimizer,
            is_train=True,
            epo=epo,
            console=console
        )

        # Run validation.
        ret_valid = run_model(
            net_pred, gade_model, to_class,
            data_loader=valid_loader,
            opt=opt,
            dct_matrices=dct_matrices,
            is_train=False,
            epo=epo,
            mode="Validation",
            console=console
        )

        # Run testing during the training loop to monitor performance on the test set.
        ret_test = run_model(
            net_pred, gade_model, to_class,
            data_loader=test_loader,
            opt=opt,
            dct_matrices=dct_matrices,
            is_train=False,
            epo=epo,
            mode="Testing",
            console=console,
            is_test=True
        )

        # Summarize and display the results for the current epoch in a table.
        results_table = Table(title=f"Epoch {epo} Summary", style="green")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Train", style="magenta")
        results_table.add_column("Validation", style="yellow")
        results_table.add_column("Test (80ms)", style="red")  # 80ms corresponds to frame #2

        results_table.add_row("MPJPE (mm)", f"{ret_train['m_p3d_h36']:.2f}",
                              f"{ret_valid['m_p3d_h36']:.2f}", f"{ret_test['#2']:.2f}")
        results_table.add_row("Loss P3D", f"{ret_train['l_p3d']:.2f}", "-", "-")
        console.print(results_table)

        # Prepare and save the log data for the current epoch to a CSV file.
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

        # Check if the current model is the best one based on validation error.
        is_best = ret_valid['m_p3d_h36'] < err_best
        if is_best:
            err_best = ret_valid['m_p3d_h36']
            console.print(
                f"[bold green]New best validation error: {err_best:.2f} mm. Saving model...[/bold green]")

        # Save the model checkpoint. `is_best` flag will determine if it overwrites 'ckpt_best.pth.tar'.
        log.save_ckpt({
            'epoch': epo,
            'lr': lr_now,
            'err': err_best,
            'state_dict': net_pred.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=is_best, opt=opt, test_dict=ret_test)


def run_model(
        net_pred: nn.Module,
        gade_model: nn.Module,
        to_class: nn.Module,
        data_loader: DataLoader,
        opt: Options,
        dct_matrices: Tuple[torch.Tensor, torch.Tensor],
        is_train: bool,
        console: Console,
        optimizer: Optional[optim.Optimizer] = None,
        epo: int = 1,
        mode: str = "Training",
        is_test: bool = False,
) -> Dict[str, float]:
    """
    Executes a single epoch of training, validation, or testing.

    Args:
        net_pred: The primary pose prediction model (EKAE).
        gade_model: The auxiliary motion prediction model (GADE).
        to_class: The action classification model (GCN).
        data_loader: The DataLoader for the current dataset split.
        opt: Configuration options.
        dct_matrices: A tuple containing pre-computed DCT and IDCT matrices.
        is_train: A boolean flag, True for training mode, False otherwise.
        console: The rich console instance for progress bar display.
        optimizer: The optimizer for training. Required if `is_train` is True.
        epo: The current epoch number (for display purposes).
        mode: A string indicating the current mode ("Training", "Validation", "Testing").
        is_test: A flag to indicate if this is a test run during a training loop.

    Returns:
        A dictionary containing the calculated metrics for the epoch.
    """
    # --- Set model to the correct mode (train or evaluation) ---
    if is_train:
        net_pred.train()
        gade_model.train()
        to_class.train()
    else:
        net_pred.eval()
        gade_model.eval()
        to_class.eval()

    # --- Initialize variables for accumulating metrics ---
    total_loss_p3d = 0  # Total combined loss during training.
    total_mpjpe_p3d_h36 = 0  # Total Mean Per Joint Position Error.
    total_samples = 0  # Total number of samples processed.

    # The prediction length (`output_n`) is 25 for testing, and 10 for train/val.
    real_output_n = opt.output_n if not is_test else 25
    # `ITERA` set as model-specific parameter for iterative refinement.
    ITERA = 1 if not is_test else 3

    # For testing, we need to store per-frame errors.
    if is_test:
        per_frame_mpjpe = np.zeros(real_output_n)

    # Unpack the DCT and IDCT matrices.
    dct_m, idct_m = dct_matrices

    # --- Configure the rich progress bar for this run ---
    progress_columns = [
        TextColumn(f"[bold cyan]{mode} Epoch {epo}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[magenta]{task.fields[info]}"),
    ]

    # The main loop iterates over the data provided by the DataLoader.
    with Progress(*progress_columns, console=console, transient=True) as progress:
        task = progress.add_task("Processing...", total=len(data_loader), info="")

        for i, (p3d_h36, class_gt) in enumerate(data_loader):
            batch_size = p3d_h36.shape[0]

            # Skip batches with a single sample during training to avoid issues with batch normalization.
            if batch_size == 1 and is_train:
                continue

            # Move input data to the specified compute device (e.g., GPU).
            p3d_h36 = p3d_h36.float().to(opt.device)

            # --- Forward Pass ---
            # Prepare model inputs from the full sequence.
            p3d_src = p3d_h36.clone()[:, :, DIM_USED]  # Select relevant dimensions.
            # `p3d_sup` is the ground truth for the supervision signal.
            p3d_sup = p3d_src.clone()[:, -real_output_n - opt.kernel_size:].reshape(
                -1, opt.kernel_size + real_output_n, len(DIM_USED) // 3, 3
            )

            # The main model `net_pred` makes the prediction.
            p3d_out_all = net_pred(p3d_src, input_n=opt.input_n, output_n=10, itera=ITERA)

            # Post-process the output of `net_pred` for the testing protocol.
            if is_test:
                p3d_out_all = p3d_out_all[:, opt.kernel_size:].transpose(1, 2)
                p3d_out_all = p3d_out_all.reshape([batch_size, 10 * ITERA, -1])[:, :real_output_n]

            # The auxiliary models are only used during training for loss calculation.
            if is_train:
                class_gt = class_gt.long().to(opt.device)
                # Prepare input for GADE: convert to frequency domain using DCT.
                motion_input = p3d_src[:, :opt.input_n, :]
                motion_input_dct = torch.matmul(dct_m, motion_input)
                # Get prediction from GADE and convert back to time domain.
                motion_pred_dct = gade_model(motion_input_dct)
                motion_pred = torch.matmul(idct_m.unsqueeze(0), motion_pred_dct)[:, -2 * real_output_n:, :]
                # Use the GADE prediction for action classification.
                motion_class_in = motion_pred.reshape(-1, 2 * real_output_n * opt.in_features)
                motion_class_out = to_class(motion_class_in)
                motion_out_all = motion_pred.reshape(batch_size, opt.kernel_size + real_output_n, 1,
                                                     len(DIM_USED) // 3, 3)

            # Reconstruct the full 3D pose from the model's prediction.
            p3d_out = p3d_h36.clone()[:, opt.input_n:opt.input_n + real_output_n]
            # Fill the predicted values into the used dimensions.
            p3d_out[:, :, DIM_USED] = p3d_out_all[:, opt.kernel_size:, 0] if not is_test else p3d_out_all
            # Apply the joint correction rule.
            p3d_out[:, :, INDEX_TO_IGNORE] = p3d_out[:, :, INDEX_TO_EQUAL]
            p3d_out = p3d_out.reshape(-1, real_output_n, 32, 3)
            # Reshape ground truth for loss/metric calculation.
            p3d_h36_reshaped = p3d_h36.reshape(-1, opt.input_n + real_output_n, 32, 3)

            # --- Training Step (if in training mode) ---
            if is_train:
                # Reshape `p3d_out_all` for loss calculation.
                p3d_out_all = p3d_out_all.reshape([batch_size, opt.kernel_size + real_output_n, len(DIM_USED) // 3, 3])
                # Loss from the main model (EKAE).
                loss_p3d = torch.mean(torch.norm(p3d_out_all - p3d_sup, dim=3))
                # Loss from the auxiliary motion model (GADE).
                loss_motion = torch.mean(torch.norm(motion_out_all[:, :, 0] - p3d_sup, dim=3))
                # Loss from the action classification model.
                loss_class = F.cross_entropy(motion_class_out, class_gt, reduction='mean')

                # Combine the three losses with specified weights.
                total_loss = ALPHA * loss_p3d + BETA * loss_motion + GAMMA * loss_class

                # Standard backpropagation steps.
                optimizer.zero_grad()
                total_loss.requires_grad_(True)
                total_loss.backward()
                # Clip gradients to prevent exploding gradients.
                nn.utils.clip_grad_norm_(
                    list(gade_model.parameters()),
                    max_norm=opt.max_norm
                )
                optimizer.step()

                # Accumulate loss and update the progress bar.
                total_loss_p3d += total_loss.item() * batch_size
                progress.update(task, advance=1, info=f"Loss: {total_loss.item():.4f}")
            else:
                # If not training, just advance the progress bar.
                progress.update(task, advance=1)

            # --- Evaluation Metric Calculation (MPJPE) ---
            if not is_test:
                # For train/val, calculate the overall MPJPE.
                mpjpe = torch.mean(
                    torch.norm(p3d_h36_reshaped[:, opt.input_n:opt.input_n + real_output_n] - p3d_out, dim=3))
                total_mpjpe_p3d_h36 += mpjpe.item() * batch_size
            else:
                # For testing, calculate MPJPE for each future frame separately.
                normalized_p3d = torch.norm(p3d_h36_reshaped[:, opt.input_n:] - p3d_out, dim=3)
                mpjpe_per_frame = torch.sum(torch.mean(normalized_p3d, dim=2), dim=0)
                per_frame_mpjpe += mpjpe_per_frame.cpu().data.numpy()

            # Increment the sample counter.
            total_samples += batch_size

    # --- Aggregate and Return Results ---
    ret = {}
    if is_train:
        # Return the average training loss.
        ret["l_p3d"] = total_loss_p3d / total_samples

    if is_test:
        # For testing, return the per-frame MPJPE.
        per_frame_mpjpe /= total_samples
        for j in range(real_output_n):
            ret[f"#{j + 1}"] = per_frame_mpjpe[j]
    else:
        # For train/val, return the overall average MPJPE.
        ret["m_p3d_h36"] = total_mpjpe_p3d_h36 / total_samples

    return ret


if __name__ == '__main__':
    # This block is the entry point of the script.
    # It parses command-line arguments and calls the main function.
    option = Options().parse()
    main(option)
