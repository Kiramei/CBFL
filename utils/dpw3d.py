import pickle as pkl
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
# --- Use rich for beautiful console output ---
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from torch.utils.data import Dataset

from utils import ang2joint


class Datasets(Dataset):
    """
    Dataset class for the 3D Poses in the Wild (3DPW) dataset.
    Handles loading, processing, and preparing 3D human pose data from .pkl files.
    """

    # Define base path and split directories as class attributes for clarity.
    BASE_PATH = Path("./datasets/3dpw/")
    SPLIT_DIRS = {
        0: 'train',
        1: 'validation',
        2: 'test',
    }
    SKELETON_PATH = './body_models/smpl_skeleton.npz'

    def __init__(self, opt, split: int = 0):
        """
        Initializes the 3DPW dataset.

        Args:
            opt (Options): An object containing configuration parameters like input_n, output_n.
            split (int, optional): Data split. 0 for train, 1 for validation, 2 for test. Defaults to 0.
        """
        self.split = split
        self.in_n = opt.input_n
        self.out_n = opt.output_n
        self.skip_rate = opt.skip_rate

        # Data storage
        self.p3d: Dict[int, np.ndarray] = {}  # Stores processed 3D pose sequences.
        self.data_idx: List[Tuple[int, int]] = []  # Maps item index to (sequence_key, start_frame).
        self.motion_class_labels: List[int] = []  # Stores class label for each sequence.

        # Use the device specified in options, fallback to 'cpu' if not available.
        # This is for the ang2joint conversion, which was originally on CUDA.
        self.device = opt.device if torch.cuda.is_available() else 'cpu'

        # Load skeleton information once.
        self.p3d0, self.parents = self._load_skeleton()

        # Load and process all data files for the specified split.
        self._load_data()

    def _load_skeleton(self) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        Loads the SMPL skeleton template and parent information.

        Returns:
            Tuple[torch.Tensor, Dict[int, int]]: A tuple containing:
                - The template skeleton pose (p3d0).
                - A dictionary mapping each joint to its parent.
        """
        skel = np.load(self.SKELETON_PATH)
        p3d0 = torch.from_numpy(skel['p3d0']).float().to(self.device)[:, :22]

        parents_array = skel['parents']
        parents_dict = {i: parents_array[i] for i in range(22)}  # Only need parents for the first 22 joints.

        return p3d0, parents_dict

    def _process_file(self, file_path: Path, class_map: Dict[str, int]) -> List[Tuple[np.ndarray, int]]:
        """
        Loads and processes a single .pkl file.

        Args:
            file_path (Path): The path to the .pkl file.
            class_map (Dict[str, int]): A mapping from action class names to integer labels.

        Returns:
            List[Tuple[np.ndarray, int]]: A list where each tuple contains a processed
                                          motion sequence (as a numpy array) and its class label.
        """
        with open(file_path, 'rb') as f:
            data = pkl.load(f, encoding='latin1')

        # The key 'poses_60Hz' contains a list of motion sequences within a single file.
        # We treat each of these as a separate data sequence.
        processed_sequences = []

        # Extract the action class from the filename.
        # e.g., "downtown_bus_00.pkl" -> "downtown_bus"
        class_name = file_path.stem.rsplit('_', 1)[0]
        class_label = class_map[class_name]

        # Define the downsampling rate to convert from 60Hz to 25Hz.
        sample_rate = 60 // 25

        for poses_60hz in data['poses_60Hz']:
            num_frames_60hz = poses_60hz.shape[0]

            # Subsample the sequence from 60Hz to 25Hz.
            fidxs = range(0, num_frames_60hz, sample_rate)
            poses_25hz = poses_60hz[fidxs]

            num_frames_25hz = poses_25hz.shape[0]

            # Convert pose parameters (angles) to 3D joint positions.
            poses_tensor = torch.from_numpy(poses_25hz).float().to(self.device)
            poses_tensor = poses_tensor.view(num_frames_25hz, -1, 3)
            poses_tensor = poses_tensor[:, :-2]  # Exclude the last 2 joints.
            poses_tensor[:, 0] = 0  # Remove global rotation.

            p3d0_tmp = self.p3d0.repeat([num_frames_25hz, 1, 1])
            p3d = ang2joint.ang2joint(p3d0_tmp, poses_tensor, self.parents)

            p3d_numpy = p3d.cpu().numpy()
            processed_sequences.append((p3d_numpy, class_label))

        return processed_sequences

    def _load_data(self):
        """
        Main data loading loop. Finds all .pkl files, processes them, and creates
        sliding window samples. Displays a progress bar.
        """
        data_path = self.BASE_PATH / self.SPLIT_DIRS[self.split]

        # Find all .pkl files in the directory.
        files = sorted(list(data_path.glob("*.pkl")))

        # Create a mapping from unique action class names to integer labels.
        class_names = sorted(list(set(f.stem.rsplit('_', 1)[0] for f in files)))
        class_map = {name: i for i, name in enumerate(class_names)}

        seq_len = self.in_n + self.out_n
        key_counter = 0

        # Setup rich progress bar.
        progress_columns = [
            TextColumn(f"[bold cyan]Loading {self.SPLIT_DIRS[self.split].capitalize()} Data[/bold cyan]"),
            BarColumn(bar_width=None), MofNCompleteColumn(), TimeElapsedColumn(),
            TextColumn("[green]{task.description}"),
        ]

        with Progress(*progress_columns, transient=False) as progress:
            task = progress.add_task("Initializing...", total=len(files))

            for file_path in files:
                progress.update(task, advance=1, description=f"{file_path.name}")

                # Each file can contain multiple motion sequences.
                sequences_in_file = self._process_file(file_path, class_map)

                for p3d_data, class_label in sequences_in_file:
                    num_frames = p3d_data.shape[0]

                    # Store the processed data and its label.
                    self.p3d[key_counter] = p3d_data
                    self.motion_class_labels.append(class_label)

                    # Create sliding window samples.
                    # Test split uses a stride of 1, train/val use skip_rate.
                    stride = 1 if self.split == 2 else self.skip_rate
                    valid_frames = np.arange(0, num_frames - seq_len + 1, stride)

                    self.data_idx.extend([(key_counter, start) for start in valid_frames])
                    key_counter += 1

            progress.update(task, description=f"Loaded {len(files)} files, found {key_counter} sequences.")

    def __len__(self) -> int:
        """Returns the total number of samples (sliding windows) in the dataset."""
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
        # Retrieve the sequence key and the start frame for this item.
        sequence_key, start_frame = self.data_idx[item]

        # Define the full frame range for the sample.
        frame_indices = np.arange(start_frame, start_frame + self.in_n + self.out_n)

        # Get the motion data and the class label.
        motion_data = self.p3d[sequence_key][frame_indices]
        motion_label = self.motion_class_labels[sequence_key]

        return motion_data, motion_label
