import torch
import torch.nn as nn
import numpy as np
torch.backends.cudnn.deterministic = True

class bModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = self.sig(x)
        return x

class fModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = self.sig(x)
        return x

b = bModel().double()
f = fModel().double()
b.eval()
f.eval()

with torch.no_grad():
    x = torch.rand((1, 1, 100, 100)).double()
    # x, b, f = x.to("cuda"), b.to("cuda"), f.to("cuda")
    print(torch.allclose(b(x).flatten(), f(x).flatten()))
    print((b(x).squeeze() - f(x).squeeze()).norm())