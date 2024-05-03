"""
PyTorch Dataset for Dex-Net 2.0 and Dex-Net 3.0 datasets.
"""

import time
import glob
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

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
    POSE_IDX = [3, 4]

    normalizers = (
        0.59247871,
        0.06873399,
        0.35668945,
        0.23767089,
        0.00296211,
        1.46679687
    )  # mean, std value for depth images, z-pose, angle-pose (from analyze.py script)
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
        self.poses = self._get_data(self.KEYS["pose"])[:, self.POSE_IDX].squeeze()

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
        self, img: torch.Tensor, pose: torch.Tensor
    ):
        """
        Preprocesses the image for training.
        """
        assert len(img.shape) == 3, "Image shape must be (H, W, C)."

        img = img.permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)

        if self.transform:
            # multipicative denoising
            gamma_dist = torch.distributions.Gamma(1000, 1000)
            sample = gamma_dist.sample()
            multipliers = torch.tile(sample, (1, 32, 32))
            img = img * multipliers


            if np.random.rand() < 0.5:
                # Create noise -- with same shape as image.
                gp_noise = torch.randn(8, 8) * 0.005
                gp_noise = F.interpolate(
                    gp_noise.unsqueeze(0).unsqueeze(0), scale_factor=4.0, mode="bicubic"
                ).squeeze().unsqueeze(0)
                assert gp_noise.shape == img.shape, "Noise shape mismatch."

                # Add the noise to the image where pixel values are greater than 0
                img[img > 0] += gp_noise[img > 0].float()
            
            flip = 1
            # Image augmentation.
            if np.random.rand() < 0.5:
                img = transforms.functional.rotate(img, 180)
                flip *= -1
            if np.random.rand() < 0.5:
                img = transforms.functional.vflip(img)
                flip *= -1
            if np.random.rand() < 0.5:
                img = transforms.functional.hflip(img)
            
            if len(pose) == 2:
                pose[1] *= flip

        # Normalize image data
        img = (img - self.normalizers[0]) / self.normalizers[1]

        return img, pose

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

        if len(self.POSE_IDX) == 1:
            pose = ((self.poses[idx] - self.normalizers[2]) / self.normalizers[3]).float().reshape(1)
        else:
            pose_z = ((self.poses[idx, 0] - self.normalizers[2]) / self.normalizers[3]).float().reshape(1)
            pose_angle = (self.poses[idx, 1] - self.normalizers[4] / self.normalizers[5]).float().reshape(1)
            pose = torch.tensor([pose_z, pose_angle])
        

        depth_im, pose = self.preprocess(depth_im, pose)

        # return depth_im, grasp_metric, pose
        return depth_im, grasp_metric, pose



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
        .0325622,
        3.0742,
        1.9395
    )  # mean, std value for depth images, z pose (from analyze.py script)
    threshold = .2
    # The dataset actually thresholds at .002, so we multiply by 100 for consistency between the datasets
    metric_multiplier = 100

    # we only need z for parallel jaw since the anlge is encoded in the image rotation
    POSE_IDX = [3]

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
