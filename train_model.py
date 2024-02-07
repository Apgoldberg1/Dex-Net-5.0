from torch_dataset import Dex3Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from copy import copy
from grasp_model import DexNet3 as Model
import os
#from grasp_model import ResNet18 as Model



def train(config):
    save_name, save_directory = config["outputs"]["save_name"], config["outputs"]["save_directory"]
    batch_size = config["training"]["batch_size"]
    for epoch in range(config["training"]["num_epochs"]):
        tot_loss, tot_preds = 0.0, 0
        for i, batch in enumerate(train_loader):
            depth_ims, pose, wrench_resistances = batch
            depth_ims, pose, wrench_resistances = depth_ims.to(device), pose.to(device), wrench_resistances.to(device)

            optimizer.zero_grad()

            outs = model(depth_ims, pose)

            loss = criterion(outs, (wrench_resistances >= .2).float())

            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            tot_preds += len(pose)

            train_print_freq, val_print_freq = config["outputs"]["training_print_every"], config["outputs"]["val_print_every"]
            if i % train_print_freq == train_print_freq - 1:
                print("train_loss:", tot_loss / (tot_preds / batch_size))
                tot_loss, tot_preds = 0, 0

            if i % val_print_freq == val_print_freq - 1:
                loss, correct, precision, recall = eval(model, val_loader, criterion, device)
                print("validation:", loss, "correct:", correct, "precision:", precision, "recall:", recall)

        print("epoch", epoch, "complete")
        scheduler.step()
        if not os.path.exists(save_directory):
                os.makedirs(save_directory)
        torch.save(model.state_dict(), f"{save_directory}/{save_name}_in_training.pth")

        save_every_x = config["outputs"]["save_every_x_epoch"]
        if epoch % save_every_x == save_every_x - 1:
            torch.save(model.state_dict(), f"{save_directory}/epoch_{epoch}_{save_name}.pth")

    torch.save(model.state_dict(), f"{save_directory}/complete_training_{save_name}.pth")



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
            tot_preds += len(pose)

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
    from ruamel.yaml import YAML
    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser(description='Process configuration file.')
    parser.add_argument('--config', dest='config_file', metavar='CONFIG_FILE_PATH', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    path_arg = args.config_file

    config_path = Path(path_arg)
    yaml = YAML(typ='safe')
    config = yaml.load(config_path)

    dataset_path = config["training"]["dataset_path"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    batch_size = config["training"]["batch_size"]

    num_files, resize = config["training"]["num_files"], config["training"]["resize"]

    dataset = Dex3Dataset(dataset_path, preload=True, num_files=num_files, resize=resize)

    train_size = int(0.8 * len(dataset)) 
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset = copy(dataset)
    val_dataset.dataset.transform = False

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = Model()

    lr, momentum = config["optimizer"]["learning_rate"], config["optimizer"]["momentum"]
    gamma = config["optimizer"]["scheduler_gamma"]
    optimizer_name = config["optimizer"]["name"]

    if optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise AssertionError("only [adam, sgd] supported as optimizers")

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    criterion = nn.BCELoss()

    model.to(device)

    train(config)
