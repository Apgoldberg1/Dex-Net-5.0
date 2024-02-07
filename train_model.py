from torch_dataset import Dex3Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
#from grasp_model import DexNet2 as Model
from grasp_model import ResNet18 as Model


dataset_path = "dataset/dexnet_3/dexnet_09_13_17"
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Dex3Dataset(dataset_path, preload=True, num_files=2500)

train_size = int(0.8 * len(dataset)) 
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])



batch_size = 512

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


model = Model()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

num_epochs = 10

model.to(device)

###SHOULD ALSO PASS HAND_POSES 2 to the the modeel since thats the height. Also says we need channel 3 but doesnt tell us what it is
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

            if i % 100 == 99:
                print("train_loss:", tot_loss / (100 * batch_size))
                tot_loss = 0

                if i % 400 == 399:
                    loss, correct, precision_graspable = eval(model, val_loader, criterion, device)
                    print("validation:", loss, "correct:", correct, "graspable precision:", precision_graspable)

        print("epoch", epoch, "complete")

    torch.save(model.state_dict(), 'complete_training.pth')



def eval(model, val_loader, criterion, device):
    model.eval()
    model.to(device)
    tot_loss = 0.0
    correct, correct_precision, tot_graspable = 0, 0, 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            depth_ims, pose, wrench_resistances = batch
            depth_ims, pose, wrench_resistances = depth_ims.to(device), pose.to(device), wrench_resistances.to(device)

            outputs = model(depth_ims, pose)
            loss = criterion(outputs, wrench_resistances)

            tot_loss += loss.item()
            #print(outputs[0].item(), wrench_resistances[0].item())

            correct += (((outputs >= .2) & (wrench_resistances >= .2)) | 
                    ((outputs < .2) & (wrench_resistances < .2))).sum().item()

            correct_precision += ((outputs >= .2) & (wrench_resistances >= .2)).sum().item()
            tot_graspable += ((wrench_resistances >= .2)).sum().item()

    return tot_loss / (i * batch_size), correct / (i * batch_size), correct_precision / tot_graspable



if __name__=="__main__":
    train()
