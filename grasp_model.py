import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.linear1 = nn.Linear(1000, 126)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)
        self.r = nn.ReLU()

    def forward(self, x, z):
        x = self.resnet18(x)
        x = self.r(self.linear1(x))
        x = torch.concat((x, z), dim=-1)
        x = self.r(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        
        x = x.squeeze(dim=1)
        
        return x

class DexNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.relu4 = nn.ReLU()
        self.lrn = nn.LocalResponseNorm(size=5)
        self.fc = nn.Linear(3136, 1024)

        self.z_fc = nn.Linear(2, 16)
        self.z_ReLU = nn.ReLU()
    
        self.fc2 = nn.Linear(1024 + 16, 1024)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 2)

        self.softmax = nn.Softmax()

    def forward(self, x, z):
        """
        x: (batch, 1, 32, 32) depth images
        z: (batch, 2) gripper distance from camera, effector angle
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.lrn(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.lrn(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        
        z = self.z_fc(z)
        z = self.z_ReLU(z)

        x = torch.concat((x, z), dim=-1)

        x = self.fc2(x)
        x = self.relu5(x)
        x = self.fc3(x)

        x = self.softmax(x)
        return x[:, 0]

