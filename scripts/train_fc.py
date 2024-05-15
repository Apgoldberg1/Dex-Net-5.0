"""
Main training file. Contains train, eval, and config processing.
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from copy import copy
import os
import wandb


def train(config):
    save_name, save_directory = (
        config["outputs"]["save_name"],
        config["outputs"]["save_directory"],
    )
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    if config["training"]["wandb"]:
        run = wandb.init(
            project="DexNet",
            config=config,
            name=config["outputs"]["save_name"],
        )

    for epoch in range(config["training"]["num_epochs"]):
        tot_loss, tot_preds = 0.0, 0
        min_valid_loss = 99999.9
        model.train()
        for i, batch in enumerate(train_loader):
            depth_ims, wrench_resistances = batch
            depth_ims, wrench_resistances = (
                depth_ims.to(device),
                wrench_resistances.to(device),
            )
            wrench_resistances = torch.clip(wrench_resistances, 0, 1)

            outs = model(depth_ims)

            loss = criterion(outs.squeeze(), wrench_resistances.squeeze())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            tot_loss += loss.item() * len(wrench_resistances)
            tot_preds += len(wrench_resistances)

            train_print_freq, val_print_freq = (
                config["outputs"]["training_print_every"],
                config["outputs"]["val_print_every"],
            )
            if i == 0 or i % train_print_freq == train_print_freq - 1:
                print("train_loss:", tot_loss / tot_preds)
                if config["training"]["wandb"]:
                    wandb.log({"train_loss": tot_loss / tot_preds})

                tot_loss, tot_preds = 0, 0

            if i % val_print_freq == val_print_freq - 1:
                torch.save(model.state_dict(), f"{save_directory}/{save_name}_in_training.pth")
                loss = eval(
                    model, val_loader, criterion, device
                )
                print(
                    f"validation: {loss}"
                )

                if config["training"]["wandb"]:
                    wandb.log(
                        {
                            "validation": loss
                        }
                    )

                if loss < min_valid_loss:
                    torch.save(
                        model.state_dict(), f"{save_directory}/best_valid_{save_name}.pth"
                    )
                    min_valid_loss = loss

        print("epoch", epoch, "complete")
        scheduler.step()

        save_every_x = config["outputs"]["save_every_x_epoch"]
        if epoch % save_every_x == save_every_x - 1:
            torch.save(
                model.state_dict(), f"{save_directory}/epoch_{epoch}_{save_name}.pth"
            )

    torch.save(
        model.state_dict(), f"{save_directory}/complete_training_{save_name}.pth"
    )
    if config["training"]["wandb"]:
        wandb.finish()


def eval(model, val_loader, criterion, device):
    model.eval()
    model.to(device)
    tot_loss, tot_preds = 0.0, 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            depth_ims, wrench_resistances = batch
            depth_ims, wrench_resistances = (
                depth_ims.to(device),
                wrench_resistances.to(device),
            )

            outputs = model(depth_ims).squeeze()
            loss = criterion(outputs, wrench_resistances.squeeze())

            tot_loss += loss.item() * len(wrench_resistances)
            tot_preds += len(wrench_resistances)

            wrench_resistances = wrench_resistances.squeeze()

    model.train()

    return tot_loss / tot_preds

if __name__ == "__main__":
    from ruamel.yaml import YAML
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser(description="Process configuration file.")
    parser.add_argument(
        "--config",
        dest="config_file",
        metavar="CONFIG_FILE_PATH",
        type=str,
        required=True,
        help="Path to the config file",
    )
    args = parser.parse_args()
    path_arg = args.config_file

    config_path = Path(path_arg)
    yaml = YAML(typ="safe")
    config = yaml.load(config_path)

    if config["dataset"].lower() == "dexnet3":
        from dexnet.torch_dataset import Dex3Dataset as Dataset
    elif config["dataset"].lower() == "dexnet2":
        from dexnet.torch_dataset import Dex2Dataset as Dataset
    elif config["dataset"].lower() == "dexnet3fc":
        from dexnet.torch_dataset import Dex3FCDataset as Dataset
    else:
        raise AssertionError("only [dexnet3, dexnet2] supported as datasets")

    if config["model"].lower() == "dexnetbase":
        from dexnet.grasp_model import DexNetBase as Model
    elif config["model"].lower() == "efficientnet":
        from dexnet.grasp_model import EfficientNet as Model
    elif config["model"].lower() == "fcgqcnn":
        from dexnet.grasp_model import BaseFCGQCNN as Model
    else:
        raise AssertionError(
            f"{config['model']} is not a model option, try dexnet3 or resnet18 instead"
        )

    dataset_path = config["training"]["dataset_path"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    batch_size = config["training"]["batch_size"]

    dataset = Dataset(
        Path(dataset_path)
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset = copy(dataset)

    val_dataset.dataset.transform = False

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    model = Model()

    lr, momentum, weight_decay = (
        config["optimizer"]["learning_rate"],
        config["optimizer"]["momentum"],
        config["optimizer"]["weight_decay"],
    )
    gamma = config["optimizer"]["scheduler_gamma"]
    optimizer_name = config["optimizer"]["name"]

    if optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise AssertionError("only [adam, sgd] supported as optimizers")

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    criterion = nn.MSELoss() # TODO check

    model.to(device)

    train(config)
