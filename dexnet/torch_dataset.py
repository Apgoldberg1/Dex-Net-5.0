"""
Dataset for DexNet3.0 dataset.
"""

import time
import glob
from pathlib import Path
from typing import Dict, Literal, get_args, Tuple
from concurrent.futures import ThreadPoolExecutor
import tyro
import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


# DexNet3.0 dataset keys -- dataset includes {KEY}_{INDEX:5d}.npz files.
DEXNET_3_KEYS = Literal[
    "camera_intrs",
    "camera_poses",
    "collistion_free",
    "depth_ims_tf_table",
    "grasp_ids",
    "hand_poses",
    "image_labels",
    "obj_labels",
    "obj_masks",
    "pose_labels",
    "robust_suction_wrench_resistance",
]


class Dex3Dataset(Dataset):
    """
    Dataset for DexNet3.0 dataset, for suction grasp metrics.
    """
    normalizers = (
        0.59784445,
        0.00770147890625,
    )  # mean, std value for depth images (from analyze.py script)

    def __init__(self, dataset_path: Path):
        """
        Load DexNet 3.0 dataset to memory.

        Args:
            dataset_path: path to the dataset folder, contains the `.../tensors/` folder.
        """
        # Get datasets path; it must contain tensors/ folder, check if it exists.
        self.dataset_path = dataset_path
        assert (
            dataset_path / "tensors"
        ).exists(), "Dataset path does not contain tensors/ folder."

        # Dynamically calculate # of files to load, if not specified.
        self.num_files = self._get_nfiles()

        # Load data to memory.
        start = time.time()
        self.depth_im_data = self._get_data("depth_ims_tf_table")
        self.grasp_metric_data = self._get_data("robust_suction_wrench_resistance")

        self.pos_idx = self.grasp_metric_data >= .2     #TODO make this based on the parameter in the config
        print(f"Loaded data in {(time.time() - start):.2f} seconds.")

        # Misc data shape checks.
        assert (
            self.depth_im_data.shape[0]
            == self.grasp_metric_data.shape[0]
        ), "Data shapes do not match (batch size mismatch)."
        assert self.depth_im_data.shape[1:] == (
            32,
            32,
            1,
        ), "Depth image shape mismatch."

        # Calculate dataset length.
        self.dataset_len = self.depth_im_data.shape[0]

    def _get_nfiles(self) -> int:
        """Calculate number of files in the dataset, by counting files in the tensors/ folder."""
        key = get_args(DEXNET_3_KEYS)[0]
        return len(glob.glob(f"{self.dataset_path}/tensors/{key}_*.npz"))

    def _get_data(self, key: DEXNET_3_KEYS) -> torch.Tensor:
        """
        Loads data to memory, uses ThreadPoolExecutor to load data in parallel.
        Data key must be one of the predefined keys in Dexnet 3.0 dataset.
        Args:
            key: key to load data for.
        Returns:
            np.ndarray: concatenated data from all files (B, ...).
        """

        def key_to_npz(key: DEXNET_3_KEYS, idx: int) -> str:
            return f"{self.dataset_path}/tensors/{key}_{str(idx).zfill(5)}.npz"

        def loader(idx):
            with np.load(key_to_npz(key, idx)) as x:
                arr = x["arr_0"]
            return arr

        with ThreadPoolExecutor(max_workers=8) as executor:
            data = executor.map(
                lambda idx: loader(idx),
                range(self.num_files),
            )
        data = torch.from_numpy(np.concatenate(list(data)))

        return data

    def preprocess(
        self, img: torch.Tensor
    ) -> torch.Tensor:
        """
        Preprocesses the image for training.
        """
        assert len(img.shape) == 3, "Image shape must be (H, W, C)."

        img = img.permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)

        # Normalize image data.
        img = (img - self.normalizers[0]) / self.normalizers[1]

        # Create noise -- with same shape as image.
        gp_noise = torch.randn(8, 8) * 0.005
        gp_noise = F.interpolate(
            gp_noise.unsqueeze(0).unsqueeze(0), scale_factor=4.0, mode="bicubic"
        ).squeeze()
        assert gp_noise.shape[-1] == img.shape[-1], "Noise shape mismatch."

        # Add the noise to the image where pixel values are greater than 0
        mask = (img > 0).float()
        img = img + gp_noise * mask

        # Image augmentation.
        # Randomly rotate image by 180 degrees...
        if np.random.rand() < 0.5:
            img = transforms.functional.rotate(img, 180)
        # Randomly flip the image...
        if np.random.rand() < 0.5:
            img = transforms.functional.vflip(img)
        if np.random.rand() < 0.5:
            img = transforms.functional.hflip(img)

        # Resize image to 224x224.
        # img = transforms.Resize((224, 224), antialias=True)(img)  # This is expensive, too.

        return img

    def __len__(self):
        """
        Number of datapoints in the dataset.
        For Dexnet 3.0, # of data files != # of datapoints -- each file contains 1000 datapoints.
        """
        return self.dataset_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load data! Returns a dictionary with keys: depth_im, grasp_metric.
        """
        depth_im = self.depth_im_data[idx]
        grasp_metric = self.grasp_metric_data[idx]

        depth_im = self.preprocess(depth_im)

        return depth_im, grasp_metric


def testLoader():
    """
    Test Dexnet 3.0 dataset speed.
    """
    from torch.utils.data import DataLoader

    dataset_path = Path("dataset/dexnet_3/dexnet_09_13_17")
    dataset = Dex3Dataset(dataset_path)

    batch_size = 128
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Check iter speed.
    for _ in tqdm.tqdm(enumerate(dataloader)):
        continue


if __name__ == "__main__":
    tyro.cli(testLoader)
