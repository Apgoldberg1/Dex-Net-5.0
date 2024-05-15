"""
PyTorch Dataset for Dex-Net 2.0 and Dex-Net 3.0 datasets.
"""

import time
import glob
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
import tqdm
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

class Dex3FCDataset(Dataset):
    """
    Dataset for DexNet3.0 FC, where input is a full size depth image and output is a pixelwise grasp affordance map
    """
    normalizers = (
        0.59784445,
        0.00770147890625,
    )  # mean, std value for depth images (from analyze.py script). do I need this ? 
    threshold = 0.2
    metric_multiplier = 1

    def __init__(self, dataset_path: Path):
        """
        Load FC data into memory
        """
        # Get datasets path; it must contain tensors/ folder, check if it exists.
        self.transform = False # TODO try transform True
        self.dataset_path = dataset_path
        self.num_files = self._get_nfiles()

        # Load data to memory.
        start = time.time()
        self.depth_im_data = self._get_data('X')
        self.grasp_metric_data = self._get_data('Y')

        print(f"Loaded data in {(time.time() - start):.2f} seconds.")

        # Misc data shape checks.
        assert (
            self.depth_im_data.shape[0]
            == self.grasp_metric_data.shape[0]
        ), "Data shapes do not match (batch size mismatch)."

        # Calculate dataset length.
        self.dataset_len = self.depth_im_data.shape[0]

    def _get_nfiles(self) -> int:
        """Calculate number of files in the dataset, by counting files in the tensors/ folder."""
        return len(glob.glob(f"{self.dataset_path}/Y/*.npy"))

    def _get_data(self, key: str) -> torch.Tensor:
        """
        Loads data to memory, uses ThreadPoolExecutor to load data in parallel.
        Data key must be one of the predefined keys in Dexnet 3.0 dataset.
        Args:
            key: key to load data for.
        Returns:
            np.ndarray: concatenated data from all files (B, ...).
        """

        def key_to_npz(idx: int) -> str:
            if key == 'X':
                img = cv2.imread(f"{self.dataset_path}/{key}/{str(idx).zfill(4)}.png").mean(axis=2)
                # preprocess blender depth
                img = 255 - img
                kernel = np.ones((15,15), np.uint8)
                img = cv2.dilate(img, kernel, iterations=1)
                img = cv2.GaussianBlur(img, (15,15), 5)
                img = (img - img.mean()) / img.std()
                # resize and pad ? 
                return img.reshape((img.shape[0], img.shape[1], 1))
            elif key == 'Y':
                img = np.load(f"{self.dataset_path}/{key}/{str(idx)}.npy")
                img = cv2.resize(img, (241, 241)) # hardcode resizing for now
                img = img.reshape((img.shape[0], img.shape[1], 1))
                return img

        with ThreadPoolExecutor(max_workers=8) as executor:
            data = executor.map(
                lambda idx: key_to_npz(idx),
                range(self.num_files),
            )
        data = torch.from_numpy(np.array(list(data)))

        return data

    def preprocess(
        self, img: torch.Tensor
    ) -> torch.Tensor:
        """
        Preprocesses the image for training.
        """
        assert len(img.shape) == 3, "Image shape must be (H, W, C)."
        img = img.permute(2, 0, 1).float()

        if self.transform:
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
            if np.random.rand() < 0.5:
                img = transforms.functional.rotate(img, 180)
            if np.random.rand() < 0.5:
                img = transforms.functional.vflip(img)
            if np.random.rand() < 0.5:
                img = transforms.functional.hflip(img)

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
        grasp_metric = self.preprocess(grasp_metric)

        return depth_im, grasp_metric

class Dex3Dataset(Dataset):
    """
    Dataset for DexNet3.0 dataset, for suction grasp metrics.
    """
    # DexNet3.0 dataset keys -- dataset includes {KEY}_{INDEX:5d}.npz files.
    # For a full list of available keys view the README contained with the dataset
    KEYS = {
        "imgs": "depth_ims_tf_table",
        "pose": "hand_poses",
        "grasp_metric": "robust_suction_wrench_resistance",
    }

    normalizers = (
        0.59784445,
        0.00770147890625,
    )  # mean, std value for depth images (from analyze.py script)
    threshold = 0.2
    metric_multiplier = 1

    def __init__(self, dataset_path: Path):
        """
        Load DexNet 3.0 dataset to memory.

        Args:
            dataset_path: path to the dataset folder, contains the `.../tensors/` folder.
        """
        # Get datasets path; it must contain tensors/ folder, check if it exists.
        self.transform = True
        self.dataset_path = dataset_path
        assert (
            dataset_path / "tensors"
        ).exists(), "Dataset path does not contain tensors/ folder."

        self.num_files = self._get_nfiles()

        # Load data to memory.
        start = time.time()
        self.depth_im_data = self._get_data(self.KEYS["imgs"])
        self.grasp_metric_data = self._get_data(self.KEYS["grasp_metric"]).float()
        self.grasp_metric_data *= self.metric_multiplier

        self.pos_idx = self.grasp_metric_data >= self.threshold     #TODO make this based on the parameter in the config
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
        key = self.KEYS["grasp_metric"]
        return len(glob.glob(f"{self.dataset_path}/tensors/{key}_*.npz"))

    def _get_data(self, key: str) -> torch.Tensor:
        """
        Loads data to memory, uses ThreadPoolExecutor to load data in parallel.
        Data key must be one of the predefined keys in Dexnet 3.0 dataset.
        Args:
            key: key to load data for.
        Returns:
            np.ndarray: concatenated data from all files (B, ...).
        """

        def key_to_npz(key: self.KEYS, idx: int) -> str:
            return f"{self.dataset_path}/tensors/{key}_{str(idx).zfill(5)}.npz"

        def loader(idx):
            with np.load(key_to_npz(key, idx)) as x:
                arr = x["arr_0"].astype(np.float16)
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

        if self.transform:
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
            if np.random.rand() < 0.5:
                img = transforms.functional.rotate(img, 180)
            if np.random.rand() < 0.5:
                img = transforms.functional.vflip(img)
            if np.random.rand() < 0.5:
                img = transforms.functional.hflip(img)

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



class Dex2Dataset(Dex3Dataset):
    # DexNet2.0 dataset keys -- dataset includes {KEY}_{INDEX:5d}.npz files.
    # For a full list of available keys view the README contained with the dataset
    KEYS = {
        "imgs": "depth_ims_tf_table",
        "pose": "hand_poses",
        "grasp_metric": "robust_ferrari_canny",
    }
    normalizers = (
        0.7000409878366362,
        0.004000758979378677
    )  # mean, std value for depth images (from analyze.py script)
    threshold = .2
    # The dataset actually thresholds at .002, so we multiply by 100 for consistency between the datasets
    metric_multiplier = 100

def testLoader():
    """
    Test Dexnet 3.0 dataset speed.
    """
    from torch.utils.data import DataLoader

    dataset_path = Path("dataset/dexnet_3/dexnet_09_13_17")
    dataset = Dex3Dataset(dataset_path)

    batch_size = 256 
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # Check iter speed.
    for _ in tqdm.tqdm(enumerate(dataloader)):
        continue

if __name__ == "__main__":
    testLoader()
