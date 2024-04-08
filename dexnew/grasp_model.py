"""
Model for DexNet3.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
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

class fakeSuctionFCGQCNN(nn.Module):
    def __init__(self, suctionModel):
        super().__init__()
        self.gqcnn = suctionModel

    def forward(self, x, z):
        heatmap = torch.zeros_like(x)

        for i in range(x.shape[2] - 32):
            for j in range(x.shape[3] - 32):
                depth = x[:, :, i:i + 32, j:j + 32]
                heatmap[:, :, i + 16, j + 16] = self.gqcnn(depth, z)

        return heatmap

class DexNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=7, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=5, padding="same"
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding="same"
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding="same"
        )
        self.lrn = nn.LocalResponseNorm(size=2, alpha=2.0e-05, beta=0.75, k=1.0)
        self.fc = nn.Linear(16384, 1024)

        self.z_fc = nn.Linear(2, 16)

        self.fc2 = nn.Linear(1024 + 16, 1024)
        self.fc3 = nn.Linear(1024, 2)

        self.softmax = nn.Softmax()

    def forward(self, x, z):
        """
        x: (batch, 1, 32, 32) depth images
        z: (batch, 2) gripper distance from camera, effector angle
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)

        z = self.z_fc(z)
        z = self.relu(z)

        x = torch.concat((x, z), dim=-1)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        x = self.softmax(x)
        return x[:, 0]
