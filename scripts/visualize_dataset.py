import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from dexnet.torch_dataset import Dex3Dataset as DexDataset
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def main(dataset):
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    for viz_im in range(10):
        samples = np.random.choice(2000000, 25, replace=False)
        images = torch.stack([dataset[i][0] for i in samples])
        labels = torch.stack([dataset[i][1] for i in samples])

        fig, axes = plt.subplots(5, 5, figsize=(10, 10))

        for i in range(25):
            ax = axes[i // 5, i % 5]  # Determine the subplot position
            ax.imshow(images[i].permute(1, 2, 0))
            ax.axis('off')  # Turn off axis for cleaner display
            ax.text(0.5, -0.1, f"Label: {labels[i].item():.2f}", 
                    transform=ax.transAxes, ha="center")  # Add label and prediction as text at the bottom

        plt.show()
        plt.savefig(f"outputs/dataset_viz_{viz_im}")


if __name__ == "__main__":
    # path to directory containing "tensors" folder
    dataset_path = Path("../dataset/dexnet_3/dexnet_09_13_17")
    dataset = DexDataset(dataset_path)
    main(dataset)
