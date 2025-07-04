import torch
import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import Dataset

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from utils import data_utils


class Datasets(Dataset):
    """
    Dataset class for Human3.6M.
    Handles loading, processing, and preparing motion capture data for training and evaluation.
    """
    # --- Default actions and subject splits for H3.6M ---
    DEFAULT_ACTIONS = [
        "walking", "eating", "smoking", "discussion", "directions",
        "greeting", "phoning", "posing", "purchases", "sitting",
        "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"
    ]

    # Map actions to integer labels
    ACTION_LABELS = {action: i for i, action in enumerate(DEFAULT_ACTIONS)}

    # Train/Val/Test subject IDs
    SUBJECT_SPLITS = {
        0: [1, 6, 7, 8, 9],  # Training subjects
        1: [11],  # Validation subject (using subject 11 from the dataset)
        2: [5],  # Testing subject (using subject 5 from the dataset)
    }

    DATA_PATH = "./datasets/h3.6m/"

    def __init__(self, opt, actions: List[str] = None, split: int = 0):
        """
        Initializes the H3.6M dataset.

        Args:
            opt (Options): An object containing configuration parameters like input_n, output_n.
            actions (List[str], optional): A list of action names to load. Defaults to all 15 actions.
            split (int, optional): Data split. 0 for train, 1 for validation, 2 for testing. Defaults to 0.
        """
        self.split = split
        self.in_n = opt.input_n
        self.out_n = opt.output_n
        self.skip_rate = opt.skip_rate
        self.sample_rate = 2

        self.p3d: Dict[int, np.ndarray] = {}  # Stores processed 3D pose sequences, keyed by an integer
        self.data_idx: List[Tuple[int, int]] = []  # List of (sequence_key, start_frame) for __getitem__

        # This will store the class label for each sequence key
        self.motion_class_labels: List[int] = []

        # Use default actions if none are provided
        self.actions = actions if (not actions in [None, 'all']) else self.DEFAULT_ACTIONS

        # Use the device specified in options for processing, fallback to 'cuda' if not available
        self.device = opt.device if torch.cuda.is_available() else 'cpu'

        # Load data with a progress bar
        self._load_data()

    def _load_and_process_sequence(self, subject: int, action: str, subaction: int) -> Tuple[np.ndarray, int]:
        """
        Loads a single sequence file, processes it, and returns the 3D poses.

        Args:
            subject (int): Subject ID.
            action (str): Action name.
            subaction (int): Subaction ID.

        Returns:
            Tuple[np.ndarray, int]: A tuple containing the processed 3D pose data and the number of frames.
        """
        filename = f'{self.DATA_PATH}/S{subject}/{action}_{subaction}.txt'

        sequence_expmap = data_utils.readCSVasFloat(filename)
        num_frames_original, _ = sequence_expmap.shape

        # Subsample frames
        sampled_indices = range(0, num_frames_original, self.sample_rate)
        sequence_expmap = sequence_expmap[sampled_indices, :]
        num_frames_sampled = len(sampled_indices)

        # Process to 3D coordinates
        # Note: Moving data to GPU for a single transform is inefficient,
        # but we keep it to maintain original functionality.
        sequence_tensor = torch.from_numpy(sequence_expmap).float().to(self.device)

        # Remove global rotation and translation
        sequence_tensor[:, 0:6] = 0

        p3d_tensor = data_utils.expmap2xyz_torch(sequence_tensor)
        p3d_numpy = p3d_tensor.view(num_frames_sampled, -1).cpu().numpy()

        return p3d_numpy, num_frames_sampled

    def _load_data(self):
        """
        Main data loading loop. Iterates through subjects, actions, and subactions
        to populate the dataset. Displays a progress bar.
        """
        seq_len = self.in_n + self.out_n
        subjects = self.SUBJECT_SPLITS[self.split]

        # --- Prepare a list of all files to be loaded for the progress bar ---
        tasks_to_load = []
        if self.split < 2:  # Train and validation splits
            for subj in subjects:
                for action in self.actions:
                    for subact in [1, 2]:
                        tasks_to_load.append((subj, action, subact))
        else:  # Test split, special handling
            for subj in subjects:
                for action in self.actions:
                    # Both subactions are loaded for each action in the test protocol
                    tasks_to_load.append((subj, action, 1))
                    tasks_to_load.append((subj, action, 2))

        # --- Setup rich progress bar ---
        progress_columns = [
            TextColumn(f"[bold cyan]Loading {['Train', 'Val', 'Test'][self.split]} Data[/bold cyan]"),
            BarColumn(bar_width=None), MofNCompleteColumn(), TimeElapsedColumn(),
            TextColumn("[green]{task.description}"),
        ]

        with Progress(*progress_columns, transient=False) as progress:
            task = progress.add_task("Initializing...", total=len(tasks_to_load))

            key_counter = 0
            # --- Main data loading and processing loop ---
            if self.split < 2:  # Train/Validation logic
                for subj, action, subact in tasks_to_load:
                    progress.update(task, advance=1, description=f"S{subj}/{action}_{subact}.txt")

                    p3d_data, num_frames = self._load_and_process_sequence(subj, action, subact)

                    self.p3d[key_counter] = p3d_data
                    self.motion_class_labels.append(self.ACTION_LABELS[action])

                    # Create sliding window samples
                    valid_frames = np.arange(0, num_frames - seq_len + 1, self.skip_rate)
                    self.data_idx.extend([(key_counter, start) for start in valid_frames])
                    key_counter += 1

            else:  # Test logic
                # Group tasks by action to handle pairs of subactions
                from itertools import groupby

                # We need to process subaction 1 and 2 for each action together
                action_groups = groupby(tasks_to_load, key=lambda x: (x[0], x[1]))

                for (subj, action), group in action_groups:
                    # Process subaction 1
                    subact1_task = next(group)
                    progress.update(task, advance=1,
                                    description=f"S{subact1_task[0]}/{subact1_task[1]}_{subact1_task[2]}.txt")
                    p3d1, n_frames1 = self._load_and_process_sequence(subj, action, 1)

                    key1 = key_counter
                    self.p3d[key1] = p3d1
                    self.motion_class_labels.append(self.ACTION_LABELS[action])

                    # Process subaction 2
                    subact2_task = next(group)
                    progress.update(task, advance=1,
                                    description=f"S{subact2_task[0]}/{subact2_task[1]}_{subact2_task[2]}.txt")
                    p3d2, n_frames2 = self._load_and_process_sequence(subj, action, 2)

                    key2 = key_counter + 1
                    self.p3d[key2] = p3d2
                    self.motion_class_labels.append(self.ACTION_LABELS[action])

                    # Find valid indices using the specific 256-frame test protocol
                    fs_sel1, fs_sel2 = data_utils.find_indices_256(n_frames1, n_frames2, seq_len, input_n=self.in_n)

                    # Add indices for subaction 1
                    valid_frames1 = fs_sel1[:, 0]
                    self.data_idx.extend([(key1, start) for start in valid_frames1])

                    # Add indices for subaction 2
                    valid_frames2 = fs_sel2[:, 0]
                    self.data_idx.extend([(key2, start) for start in valid_frames2])

                    key_counter += 2

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_idx)

    def __getitem__(self, item: int) -> Tuple[np.ndarray, int]:
        """
        Retrieves a single sample from the dataset.

        Args:
            item (int): The index of the sample.

        Returns:
            Tuple[np.ndarray, int]: A tuple containing the motion sequence (input + output)
                                    and its corresponding action label.
        """
        # Retrieve the sequence key and the start frame for this item
        sequence_key, start_frame = self.data_idx[item]

        # Define the full frame range for the sample (input + output)
        frame_indices = np.arange(start_frame, start_frame + self.in_n + self.out_n)

        # Get the motion data and the class label
        motion_data = self.p3d[sequence_key][frame_indices]
        motion_label = self.motion_class_labels[sequence_key]

        return motion_data, motion_label


