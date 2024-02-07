from torch_dataset import Dex3Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from copy import copy
#from grasp_model import DexNet2 as Model
from grasp_model import ResNet18 as Model


dataset_path = "dataset/dexnet_3/dexnet_09_13_17"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
num_epochs = 30
batch_size = 4096


def train():
    for epoch in range(num_epochs):
        tot_loss = 0.0
        for i, batch in enumerate(train_loader):
            depth_ims, pose, wrench_resistances = batch
            depth_ims, pose, wrench_resistances = depth_ims.to(device), pose.to(device), wrench_resistances.to(device)

            optimizer.zero_grad()

            outs = model(depth_ims, pose)

            loss = criterion(outs, (wrench_resistances >= .2).float())

            loss.backward()
            optimizer.step()

            tot_loss += loss.item()

            if i % 50 == 49:
                print("train_loss:", tot_loss / (50 * batch_size))
                tot_loss = 0

                if i % 200 == 199:
                    loss, correct, precision, recall = eval(model, val_loader, criterion, device)
                    print("validation:", loss, "correct:", correct, "precision:", precision, "recall:", recall)

        print("epoch", epoch, "complete")
        scheduler.step()
        torch.save(model.state_dict(), 'in_training2.pth')

        if epoch % 10 == 9:
            torch.save(model.state_dict(), f"epoch_{epoch}_save2")

    torch.save(model.state_dict(), 'complete_training2.pth')



def eval(model, val_loader, criterion, device):
    model.eval()
    model.to(device)
    tot_loss, tot_preds = 0.0, 0
    correct, tot_tp, tot_fp, tot_fn = 0, 0, 0, 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            depth_ims, pose, wrench_resistances = batch
            depth_ims, pose, wrench_resistances = depth_ims.to(device), pose.to(device), wrench_resistances.to(device)

            outputs = model(depth_ims, pose)
            loss = criterion(outputs, wrench_resistances)

            tot_loss += loss.item()
            tot_preds += len(batch)

            correct += (((outputs >= .2) & (wrench_resistances >= .2)) | 
                    ((outputs < .2) & (wrench_resistances < .2))).sum().item()

            tp, fp, fn = getPrecisionRecall(outputs, wrench_resistances, thresh=.2)
            tot_tp, tot_fp, tot_fn = tot_tp + tp, tot_fp + fp, tot_fn + fn


    if tot_tp == 0:
        precision, recall = 0, 0
    else:
        precision, recall = tot_tp / (tot_tp + tot_fp), tot_tp / (tot_tp + tot_fn)

    return tot_loss / tot_preds, correct / tot_preds, precision, recall


def getPrecisionRecall(outputs, wrench_resistances, thresh=.2):
    tp = ((outputs >= thresh) & (wrench_resistances >= .2)).sum().item()
    fp = ((outputs >= thresh) & (wrench_resistances < .2)).sum().item()
    fn = ((outputs < thresh) & (wrench_resistances >= .2)).sum().item()

    return tp, fp, fn



if __name__=="__main__":
    torch.manual_seed(0)
    dataset = Dex3Dataset(dataset_path, preload=True, num_files=2500, resize=False)

    train_size = int(0.8 * len(dataset)) 
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset = copy(dataset)
    val_dataset.dataset.transform = False

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = Model()

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.90)

    criterion = nn.BCELoss()

    model.to(device)

    train()
