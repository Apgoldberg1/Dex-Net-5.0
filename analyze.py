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
    tot_correct, tot_tp, tot_fp, tot_fn = np.zeros(threshold_res), np.zeros(threshold_res), np.zeros(threshold_res), np.zeros(threshold_res)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            depth_ims, pose, wrench_resistances = batch
            depth_ims, pose, wrench_resistances = depth_ims.to(device), pose.to(device), wrench_resistances.to(device)

            outputs = model(depth_ims, pose)


            for j, thresh in enumerate(np.arange(1 / threshold_res, 1, 1 / threshold_res)):
                tot_correct[j] += (((outputs > thresh) & (wrench_resistances > .2)) | 
                    ((outputs <= thresh) & (wrench_resistances <= .2))).sum().item()

                tp, fp, fn = getPrecisionRecall(outputs, wrench_resistances, thresh=thresh)
                tot_tp[j], tot_fp[j], tot_fn[j] = tot_tp[j] + tp, tot_fp[j] + fp, tot_fn[j] + fn

            tot_preds += len(pose)

        precision, recall = np.zeros(threshold_res), np.zeros(threshold_res)
        for i in range(threshold_res):
            tp, fp, fn = tot_tp[i], tot_fp[i], tot_fn[i]

            if tp == 0:
                precision[i], recall[i] = 1, 0
            else:
                precision[i], recall[i] = tp / (tp + fp), tp / (tp + fn)

    return tot_correct / tot_preds, precision, recall

def getDatasetMeanStd(loader, device):
    tot_preds = 0
    running_im_mean, running_im_std, running_pose_2_mean, running_pose_2_std, running_pose_3_mean, running_pose_3_std = 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            depth_ims, pose, wrench_resistances = batch
            depth_ims, pose, wrench_resistances = depth_ims.to(device), pose.to(device), wrench_resistances.to(device)

            tot_preds += len(pose)

            running_im_mean += depth_ims.sum() / (32 * 32)
            running_im_std += depth_ims.std(dim=0).sum()
            running_pose_2_mean += pose[:,0].sum()
            running_pose_2_std += pose[:,0].std().sum() * len(pose)
            running_pose_3_mean += pose[:,1].sum()
            running_pose_3_std += pose[:,1].std().sum() * len(pose)
            

    return running_im_mean.item() / tot_preds, running_im_std.item() / tot_preds, running_pose_2_mean.item() / tot_preds, running_pose_2_std.item() / tot_preds, running_pose_3_mean.item() / tot_preds, running_pose_3_std.item() / tot_preds

def plotPrecisionRecall(precisions, recalls):
    plt.plot(recalls, precisions)

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    #plt.xticks(np.arange(0, 1.1, .1))
    #plt.yticks(np.arange(.4, 1.1, .1))
    #plt.ylim(.35, 1.05)
    plt.ylim(0, 1)
    
    plt.savefig("outputs/plot.jpg")

def precisionMain():
    dataset_path = "dataset/dexnet_3/dexnet_09_13_17"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model = Model()
    #model.load_state_dict(torch.load("model_zoo/epoch_19_DexNet.pth"))
    model.load_state_dict(torch.load("outputs/DexNet3/complete_training_FaithfulDexNet3_updatedaug.pth"))
    model.to(device)

    dataset = Dex3Dataset(dataset_path, preload=True, num_files=2500, resize=False)

    train_size = int(0.8 * len(dataset)) 
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = False

    batch_size = 4096

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    #train_dataset.dataset.transform = False
    #val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    correct, precisions, recalls = getAllThreshedPrecisionRecall(model, val_loader, device, threshold_res=60)

    print("correct:", correct)
    plotPrecisionRecall(precisions, recalls)

def dataStatsMain():
    dataset_path = "dataset/dexnet_3/dexnet_09_13_17"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8192

    dataset = Dex3Dataset(dataset_path, preload=True, num_files=2500, resize=False)

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(getDatasetMeanStd(loader, device))



if __name__ == "__main__":
    #dataStatsMain()
    precisionMain()
