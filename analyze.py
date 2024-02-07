from train_model import getPrecisionRecall
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_dataset import Dex3Dataset
import torch
from torch.utils.data import DataLoader, random_split
from grasp_model import DexNet3 as Model
#from grasp_model import ResNet18 as Model


def getAllThreshedPrecisionRecall(model, val_loader, device, threshold_res=10):
    model.eval()
    model.to(device)
    tot_preds = 0
    tot_correct, tot_tp, tot_fp, tot_fn = np.zeros(threshold_res + 2), np.zeros(threshold_res + 2), np.zeros(threshold_res + 2), np.zeros(threshold_res + 2)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            depth_ims, pose, wrench_resistances = batch
            depth_ims, pose, wrench_resistances = depth_ims.to(device), pose.to(device), wrench_resistances.to(device)

            outputs = model(depth_ims, pose)


            for j, thresh in enumerate(np.arange(0, 1 + (1 / threshold_res), 1 / threshold_res)):
                tot_correct[j] += (((outputs >= thresh) & (wrench_resistances >= thresh)) | 
                    ((outputs < thresh) & (wrench_resistances < thresh))).sum().item()

                tp, fp, fn = getPrecisionRecall(outputs, wrench_resistances, thresh=thresh)
                tot_tp[j], tot_fp[j], tot_fn[j] = tot_tp[j] + tp, tot_fp[j] + fp, tot_fn[j] + fn

            tot_preds += len(batch)

        precision, recall = np.zeros(threshold_res + 2), np.zeros(threshold_res + 2)
        for i in range(threshold_res + 2):
            tp, fp, fn = tot_tp[i], tot_fp[i], tot_fn[i]

            if tp == 0:
                precision[i], recall[i] = 0, 0
            else:
                precision[i], recall[i] = tp / (tp + fp), tp / (tp + fn)

    return tot_correct / tot_preds, precision, recall


def plotPrecisionRecall(precisions, recalls):
    plt.plot(recalls, precisions)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    
    plt.savefig("outputs/plot.jpg")

if __name__ == "__main__":
    dataset_path = "dataset/dexnet_3/dexnet_09_13_17"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model = Model()
    #model.load_state_dict(torch.load("model_zoo/epoch_19_DexNet.pth"))
    model.load_state_dict(torch.load("complete_training_DexNet.pth"))
    model.to(device)

    dataset = Dex3Dataset(dataset_path, preload=True, num_files=2500, resize=False)

    train_size = int(0.8 * len(dataset)) 
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = False

    batch_size = 4096

    #val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    train_dataset.dataset.transform = False
    val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    _, precisions, recalls = getAllThreshedPrecisionRecall(model, val_loader, device, threshold_res=30)

    plotPrecisionRecall(precisions, recalls)
