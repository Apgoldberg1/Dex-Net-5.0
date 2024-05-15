import torch
import torch.nn as nn
import torchvision.models as models
import torchvision

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficient_net = models.efficientnet_b0()
        self.fc = nn.Linear(1000, 2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = torchvision.transforms.functional.resize(x, (224, 224))
        x = torch.cat([x, x, x], dim=1)
        x = self.efficient_net(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x[:, 0]

class DexNetBase(nn.Module):
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
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (batch, 1, 32, 32) depth images
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
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        x = self.sigmoid(x)
        return x

class BaseFCGQCNN(nn.Module):
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
        self.conv5 = nn.Conv2d(64, 1024, (16,16), bias=True)
        self.conv6 = nn.Conv2d(1024, 1024, (1,1), bias=True)
        self.conv7 = nn.Conv2d(1024, 1, (1,1), bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (batch, 1, x, y) depth images
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

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)

        x = self.sigmoid(x)
        return x

class UNetFCGQCNN(nn.Module):
    def __init__(self):
        super(UNetFCGQCNN, self).__init__()
        # Contracting Path
        self.conv1 = self.contract_block(1, 64, 3, 1)
        self.conv2 = self.contract_block(64, 128, 3, 1)
        self.conv3 = self.contract_block(128, 256, 3, 1)
        self.conv4 = self.contract_block(256, 512, 3, 1)
        self.drop4 = nn.Dropout(0.5)
        self.conv5 = self.contract_block(512, 1024, 3, 1)
        self.drop5 = nn.Dropout(0.5)
        
        # Expansive Path
        self.up6 = self.expand_block(1024, 512)
        self.conv6 = self.contract_block_no_pool(1024, 512, 3, 1)
        self.up7 = self.expand_block(512, 256)
        self.conv7 = self.contract_block_no_pool(512, 256, 3, 1)
        self.up8 = self.expand_block(256, 128)
        self.conv8 = self.contract_block_no_pool(256, 128, 3, 1)
        self.up9 = self.expand_block(128, 64)
        self.conv9 = self.contract_block_no_pool(128, 64, 3, 1)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def __call__(self, x):
        # Contracting Path
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4_drop = self.drop4(x4)
        x5 = self.conv5(x4_drop)
        x5_drop = self.drop5(x5)
        
        # Expansive Path
        x6 = self.up6(x5_drop)
        x6 = torch.cat([x6, x4_drop], dim=1)
        x6 = self.conv6(x6)
        x7 = self.up7(x6)
        x7 = torch.cat([x7, x3], dim=1)
        x7 = self.conv7(x7)
        x8 = self.up8(x7)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = self.conv8(x8)
        x9 = self.up9(x8)
        x9 = torch.cat([x9, x1], dim=1)
        x9 = self.conv9(x9)
        
        x10 = self.final(x9)
        output = self.sigmoid(x10)
        return output

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return contract

    def contract_block_no_pool(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        return contract

    def expand_block(self, in_channels, out_channels):
        expand = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU()
        )
        return expand

class HighResFCGQCNN(BaseFCGQCNN):
    def forward(self, x):
        """
        x: (batch, 1, 32, 32) depth images
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.lrn(x)

        x1 = torch.clone(x)[:,:,:-1,:-1]
        x2 = torch.clone(x)[:,:,1:, :-1]
        x3 = torch.clone(x)[:,:,:-1, 1:]
        x4 = torch.clone(x)[:,:,1:, 1:]

        out = [0,0,0,0]
        for i, x in enumerate([x1, x2, x3, x4]):
            x = self.maxpool(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.relu(x)
            x = self.lrn(x)

            x = self.conv5(x)
            x = self.relu(x)
            x = self.conv6(x)
            x = self.relu(x)
            x = self.conv7(x)

            x = self.sigmoid(x)
            out[i] = x

        return self.interweave_images(out[0], out[1], out[2], out[3])

    def interweave_images(self, image1, image2, image3, image4):
        """
        Interweaves 4 shifted images to make one high resolution image
        image1 is left top
        image2 is right top
        image3 is left bottom
        image4 is right bottom
        """
        assert image1.shape == image2.shape == image3.shape == image4.shape, "Images must have the same shape"

        # Get the height and width of the images
        batch, channels, height, width = image1.shape

        # Interweave rows between the first two images
        interweaved_rows1 = torch.empty((batch, channels, height * 2, width), dtype=image1.dtype)
        interweaved_rows1[:, :, ::2] = image1
        interweaved_rows1[:, :, 1::2] = image2

        # Interweave rows between the second two images
        interweaved_rows2 = torch.empty((batch, channels, height * 2, width), dtype=image3.dtype)
        interweaved_rows2[:, :, ::2] = image3
        interweaved_rows2[:, :, 1::2] = image4

        # Interweave columns of the two interweaved images
        interweaved_columns = torch.empty((batch, channels, height * 2, width * 2), dtype=image1.dtype)
        interweaved_columns[:, :, :, ::2] = interweaved_rows1
        interweaved_columns[:, :, :, 1::2] = interweaved_rows2

        return interweaved_columns  

class fakeFCGQCNN(nn.Module):
    """
    Takes GQ-CNN model as input. Runs it on all 32x32 crops and returns a heatmap of grasp confidences.
    Called "fake" because it has the intended outputs of an FC-GQ-CNN, but works using for loops which is much less efficient.
    """
    def __init__(self, gqcnn_model):
        super().__init__()
        self.gqcnn = gqcnn_model

    def forward(self, x):
        heatmap = torch.zeros_like(x)

        for i in range(x.shape[2] - 31):
            for j in range(x.shape[3] - 31):
                depth = x[:, :, i:i + 32, j:j + 32]
                heatmap[:, :, i + 16, j + 16] = self.gqcnn(depth).reshape(x.shape[0], 1)

        return heatmap

# class ResNet18(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.resnet18 = models.resnet18(weights=None)
#         self.resnet18.conv1 = nn.Conv2d(
#             1, 64, kernel_size=7, stride=2, padding=3, bias=False
#         )
#         self.linear1 = nn.Linear(1000, 126)
#         self.linear2 = nn.Linear(128, 64)
#         self.linear3 = nn.Linear(64, 1)
#         self.r = nn.ReLU()

#     def forward(self, x, z):
#         x = self.resnet18(x)
#         x = self.r(self.linear1(x))
#         x = torch.concat((x, z), dim=-1)
#         x = self.r(self.linear2(x))
#         x = torch.sigmoid(self.linear3(x))

#         x = x.squeeze(dim=1)

#         return x
