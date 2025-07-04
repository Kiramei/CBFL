from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# --- Use rich for beautiful console output ---
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from utils import data_utils


class Datasets(Dataset):
    """
    Dataset class for the CMU Mocap dataset.
    Handles loading motion capture data from .txt files, processing it from exponential maps
    to 3D coordinates, and preparing it for training and evaluation.
    """

    # --- Define class constants for better organization and clarity ---
    BASE_PATH = Path("./datasets/cmu_mocap")
    DEFAULT_ACTIONS = [
        "basketball", "basketball_signal", "directing_traffic", "jumping",
        "running", "soccer", "walking", "washwindow"
    ]
    # Mapping for dataset splits to directory names.
    # Note: split 1 and 2 both use the 'test' directory but might have different sampling logic in the original design.
    SPLIT_DIRS = {0: 'train', 1: 'test', 2: 'test'}

    # Joints to ignore in the CMU dataset skeleton.
    JOINTS_TO_IGNORE = np.array([0, 1, 2, 7, 8, 13, 16, 20, 29, 24, 27, 33, 36])

    def __init__(self, opt, actions: List[str] = None, split: int = 0):
        """
        Initializes the CMU Mocap dataset.

        Args:
            opt (Options): An object containing configuration parameters like input_n, output_n.
            actions (List[str], optional): A list of action names to load. Defaults to all 8 actions.
            split (int, optional): Data split. 0 for train, 1 for test (original intent might differ), 2 for test.
        """
        self.split = split
        self.in_n = opt.input_n
        self.out_n = opt.output_n
        self.skip_rate = opt.skip_rate
        self.sample_rate = 2

        # Data storage: a dictionary mapping a unique key to a tuple of (action_label, 3d_pose_data).
        self.processed_sequences: Dict[int, Tuple[int, np.ndarray]] = {}
        # Maps an item index to (sequence_key, start_frame) for __getitem__.
        self.data_idx: List[Tuple[int, int]] = []

        # Use the device specified in options, fallback to 'cpu' if not available.
        self.device = opt.device if torch.cuda.is_available() else 'cpu'

        # Use default actions if none are provided.
        self.actions = actions if actions is not None else self.DEFAULT_ACTIONS
        self.action_map = {action: i for i, action in enumerate(self.actions)}

        # Load data with a progress bar.
        self._load_data()

        # Calculate the dimensions to use after ignoring specified joints.
        # This is kept to maintain original functionality, even if not used elsewhere in this class.
        dimensions_to_ignore = np.concatenate(
            (self.JOINTS_TO_IGNORE * 3, self.JOINTS_TO_IGNORE * 3 + 1, self.JOINTS_TO_IGNORE * 3 + 2))
        self.dimensions_to_use = np.setdiff1d(np.arange(114), dimensions_to_ignore)

    def _process_file(self, file_path: Path) -> np.ndarray:
        """
        Loads a single .txt file, processes it from expmap to 3D coordinates.

        Args:
            file_path (Path): Path to the .txt file.

        Returns:
            np.ndarray: The processed 3D pose data as a numpy array.
        """
        # Read the sequence from the text file.
        action_sequence = data_utils.readCSVasFloat(str(file_path))

        # Subsample the frames.
        num_frames_original, _ = action_sequence.shape
        even_list = range(0, num_frames_original, self.sample_rate)
        the_sequence = np.array(action_sequence[even_list, :])
        num_frames_sampled = len(even_list)

        # Convert from exponential map representation to 3D XYZ coordinates.
        expmaps_tensor = torch.from_numpy(the_sequence).float().to(self.device)
        xyz_tensor = data_utils.expmap2xyz_torch_cmu(expmaps_tensor)

        # Reshape and move to CPU as a numpy array.
        return xyz_tensor.view(num_frames_sampled, -1).cpu().numpy()

    def _load_data(self):
        """
        Main data loading loop. Finds all relevant .txt files, processes them,
        and creates sliding window samples. Displays a progress bar.
        """
        data_path = self.BASE_PATH / self.SPLIT_DIRS[self.split]
        seq_len = self.in_n + self.out_n

        # --- Prepare a list of all files to load for the progress bar ---
        files_to_process = []
        for action in self.actions:
            action_dir = data_path / action
            if action_dir.is_dir():
                # Find all .txt files for the current action.
                files = sorted(action_dir.glob("*.txt"))
                files_to_process.extend([(f, action) for f in files])

        key_counter = 0

        # --- Setup rich progress bar ---
        progress_columns = [
            TextColumn(f"[bold cyan]Loading {self.SPLIT_DIRS[self.split].capitalize()} Data[/bold cyan]"),
            BarColumn(bar_width=None), MofNCompleteColumn(), TimeElapsedColumn(),
            TextColumn("[green]{task.description}"),
        ]

        with Progress(*progress_columns, transient=False) as progress:
            task = progress.add_task("Initializing...", total=len(files_to_process))

            for file_path, action_name in files_to_process:
                progress.update(task, advance=1, description=f"{action_name}/{file_path.name}")

                # Process the file to get 3D pose data.
                processed_data = self._process_file(file_path)
                num_frames = processed_data.shape[0]
                action_label = self.action_map[action_name]

                # Store the processed data with its label.
                self.processed_sequences[key_counter] = (action_label, processed_data)

                # Create sliding window samples.
                # Note: The original code does not differentiate split 1 and 2 here,
                # so we keep that behavior. A stride of `skip_rate` is used for all splits.
                valid_frames = np.arange(0, num_frames - seq_len + 1, self.skip_rate)
                self.data_idx.extend([(key_counter, start) for start in valid_frames])

                key_counter += 1

            progress.update(task, description=f"Loaded {len(files_to_process)} files.")

    def __len__(self) -> int:
        """Returns the total number of samples (sliding windows) in the dataset."""
        return len(self.data_idx)

    def __getitem__(self, item: int) -> Tuple[int, np.ndarray]:
        """
        Retrieves a single sample from the dataset.

        Args:
            item (int): The index of the sample.

        Returns:
            Tuple[int, np.ndarray]: A tuple containing the action label and the motion sequence slice.
        """
        # Retrieve the sequence key and the start frame for this item.
        sequence_key, start_frame = self.data_idx[item]

        # Define the full frame range for the sample.
        frame_indices = np.arange(start_frame, start_frame + self.in_n + self.out_n)

        # Get the action label and the full sequence data from storage.
        action_label, full_sequence = self.processed_sequences[sequence_key]

        # Slice the sequence to get the specific sample window.
        sequence_slice = full_sequence[frame_indices]

        return action_label, sequence_slice
