import torch
import numpy as np
import matplotlib.pyplot as plt
from dexnew.torch_dataset import Dex3Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from dexnew.grasp_model import DexNet3 as Model


def save_ims(model, loader, loaderNoNormal, device, num_imgs=1):
    saved = 0
    loaderNoNormal = iter(loaderNoNormal)

    import os
    os.makedirs("outputs", exist_ok=True)

    for i, batch in enumerate(loader):
        batchNoNormal = next(loaderNoNormal)
        if saved >= num_imgs:
            break
        depth_ims, pose, wrench_resistances = batch
        for j, GT in enumerate(wrench_resistances):
            if GT > 0.2:
                break
        else:
            continue

        depth_ims, pose, wrench_resistances = (
            depth_ims.to(device),
            pose.to(device),
            wrench_resistances.to(device),
        )

        outs = model(depth_ims, pose)
        depth_ims, pose, wrench_resistances = (
            depth_ims.to("cpu"),
            pose.to("cpu"),
            wrench_resistances.to("cpu"),
        )
        # depth_ims, pose, wrench_resistances = batchNoNormal

        to_show = depth_ims[j]
        to_show = to_show.reshape(32, 32)

        plt.imshow(to_show, cmap="gray")
        plt.xlabel(f"Pred: {outs[j]:.3f} GT: {wrench_resistances[j]:.3f}")
        plt.savefig(f"outputs/img_true{i}.jpg")
        print(i)
        saved += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process configuration file.")
    parser.add_argument(
        "--model_file",
        dest="model_file",
        metavar="MODEL_PATH",
        type=str,
        required=True,
        help="Path to model file",
    )

    args = parser.parse_args()
    model_path = args.model_file

    dataset_path = "dataset/dexnet_3/dexnet_09_13_17"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    resize = False
    dataset = Dex3Dataset(dataset_path, preload=True, num_files=25, resize=resize)
    datasetNoNormal = Dex3Dataset(
        dataset_path, preload=True, num_files=25, resize=resize
    )
    datasetNoNormal.normalize = False

    batch_size = 256
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    loaderNoNormal = DataLoader(
        dataset=datasetNoNormal, batch_size=batch_size, shuffle=False
    )

    save_ims(model, loader, loaderNoNormal, device, num_imgs=5)
