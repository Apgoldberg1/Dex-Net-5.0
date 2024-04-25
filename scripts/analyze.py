from train_model import getPrecisionRecall
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from copy import copy
import os
from pathlib import Path


def getAllThreshedPrecisionRecall(model, val_loader, device, gt_thresh, threshold_res=10):
    """
    returns 3 arrays: accuracies, precisions, and recalls at each threshold value uniformly distributed between 0 and 1.
    """

    model.eval()
    model.to(device)
    tot_preds = 0
    tot_correct, tot_tp, tot_fp, tot_fn = (
        np.zeros(threshold_res),
        np.zeros(threshold_res),
        np.zeros(threshold_res),
        np.zeros(threshold_res),
    )

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            depth_ims, wrench_resistances = batch
            depth_ims, wrench_resistances = (
                depth_ims.to(device),
                wrench_resistances.to(device),
            )

            outputs = model(depth_ims).reshape(len(wrench_resistances), 1)
            wrench_resistances = wrench_resistances.reshape(len(wrench_resistances), 1)
            assert outputs.shape == wrench_resistances.shape, "shape mismatch between gt and model outputs"

            for j, thresh in enumerate(
                np.arange(1 / threshold_res, 1, 1 / threshold_res)
            ):
                tot_correct[j] += (
                    (
                        ((outputs > thresh) & (wrench_resistances > gt_thresh))
                        | ((outputs <= thresh) & (wrench_resistances <= gt_thresh))
                    )
                    .sum()
                    .item()
                )

                tp, fp, fn = getPrecisionRecall(
                    outputs, wrench_resistances, gt_thresh, thresh=thresh
                )
                tot_tp[j], tot_fp[j], tot_fn[j] = (
                    tot_tp[j] + tp,
                    tot_fp[j] + fp,
                    tot_fn[j] + fn,
                )

            tot_preds += len(wrench_resistances)

        precision, recall = np.zeros(threshold_res), np.zeros(threshold_res)
        for i in range(threshold_res):
            tp, fp, fn = tot_tp[i], tot_fp[i], tot_fn[i]

            if tp == 0:
                precision[i], recall[i] = 1, 0
            else:
                precision[i], recall[i] = tp / (tp + fp), tp / (tp + fn)

    return tot_correct / tot_preds, precision, recall


def getDatasetMeanStd(loader, device, metric_thresh):
    tot_preds = 0
    running_im_mean, running_im_std, running_correct = 0, 0, 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            depth_ims, wrench_resistances = batch
            depth_ims, wrench_resistances = (
                depth_ims.to(device),
                wrench_resistances.to(device),
            )

            tot_preds += len(wrench_resistances)

            running_im_mean += depth_ims.sum() / (32 * 32)
            running_im_std += depth_ims.std(dim=0).sum()
            running_correct += (wrench_resistances > metric_thresh).sum()

            if tot_preds > 1_000_000:
                break

    return (
        running_im_mean.item() / tot_preds,
        running_im_std.item() / tot_preds,
        running_correct.item() / tot_preds
    )


def plotPrecisionRecall(precisions, recalls):
    """
    Takes in precision and recall arrays and saves a precision recall curve to outputs/plot.jpg
    """
    plt.plot(recalls, precisions)

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.xticks(np.arange(0, 1.1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    save_dir = "outputs"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(f"{save_dir}/precision_recall_plot.jpg")


def precisionMain(model_path, dataset_path, gt_thresh, ordered_split=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    dataset = Dataset(dataset_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    batch_size = 64

    if ordered_split:
        val_sampler = SubsetRandomSampler(torch.arange(0, val_size))

        val_dataset = copy(dataset)
        val_dataset.transform = False

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=4,
            sampler=val_sampler,
        )
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        val_dataset.dataset.transform = False

        val_loader = DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

    correct, precisions, recalls = getAllThreshedPrecisionRecall(
        model, val_loader, device, gt_thresh, threshold_res=80
    )

    print("accuracies:", correct)
    print("precisions:", precisions)
    print("recalls:", recalls)
    plotPrecisionRecall(precisions, recalls)


def dataStatsMain(dataset_path, metric_thresh):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8192

    dataset = Dataset(dataset_path)

    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    print(getDatasetMeanStd(loader, device, metric_thresh))


def getModelSummary():
    from torchinfo import summary

    model = Model()
    summary(model, input_size=[(64, 1, 32, 32)])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process configuration file.")
    parser.add_argument(
        "--model_path",
        dest="model_path",
        metavar="MODEL_PATH",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        metavar="MODEL_NAME",
        type=str,
        required=True,
        help="name of model eg EfficientNet, DexNetBase",
    )
    args = parser.parse_args()

    model_path = args.model_path
    model_name = args.model_name

    if model_name.lower() == "dexnetbase":
        from dexnet.grasp_model import DexNetBase as Model
    elif model_name.lower() == "efficientnet":
        from dexnet.grasp_model import EfficientNet as Model
    else:
        raise AssertionError(f"{model_name} as model_name arg is not supported")

    getModelSummary()

    dataset_path = Path("dataset/dexnet_3/dexnet_09_13_17")
    from dexnet.torch_dataset import Dex3Dataset as Dataset
    gt_thresh = .2

    """
    Uncomment for Dex2Dataset and parallel jaw analysis
    """
    # from dexnet.torch_dataset import Dex2Dataset as Dataset
    # dataset_path = Path("dataset/dexnet_2/dexnet_2_tensor")
    # gt_thresh = .2

    # print("Getting data statistics")
    # dataStatsMain(dataset_path, .2)

    print("Creating precision recall curve")
    precisionMain(model_path, dataset_path, gt_thresh, ordered_split=True)
