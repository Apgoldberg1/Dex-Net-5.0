import torch
import torch.nn as nn
import numpy as np
torch.backends.cudnn.deterministic = True

class BModel(nn.Module):
    def __init__(self, gq):
        super().__init__()
        self.gq = gq
    def forward(self, x):
        out = torch.zeros((x.shape[0], 1, x.shape[2] - 100 + 1, x.shape[3] - 100 + 1))
        for w in range(x.shape[2] - 100 + 1):
            for h in range(x.shape[3] - 100 + 1):
                out[:, :, w, h] = self.gq(x[:, :, w:w+100, h:h+100])
        return out

class bModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10000, 1000)
        self.sig = nn.Sigmoid()
        self.fc2 = nn.Linear(1000, 1)
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.sig(x)
        x = self.fc2(x)
        return x

class fModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1000, (100,100), bias=True)
        self.sig = nn.Sigmoid()
        self.conv2 = nn.Conv2d(1000, 1, (1,1), bias=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.sig(x)
        x = self.conv2(x)
        return x

b = bModel()
f = fModel()

f_w = f.state_dict()
f_w['conv1.weight'] = b.state_dict()["fc1.weight"].reshape(1000, 1, 100, 100)
f_w['conv1.bias'] = b.state_dict()["fc1.bias"]
f_w['conv2.weight'] = b.state_dict()["fc2.weight"].reshape(1, 1000, 1, 1)
f_w['conv2.bias'] = b.state_dict()["fc2.bias"]
f.load_state_dict(f_w)
b = BModel(b)

with torch.no_grad():
    x = 100 * torch.rand((1, 1, 101, 101))
    out_b = b(x).squeeze()
    out_f = f(x).squeeze()
    x, b, f = x.to("cuda"), b.to("cuda"), f.to("cuda")
    print(out_b.shape)
    print(out_f.shape)
    print(torch.allclose(out_b.flatten(), out_f.flatten()))
    print(out_b - out_f)