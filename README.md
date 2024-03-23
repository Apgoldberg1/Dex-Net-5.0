# DexNew - A PyTorch implementation to train on the Dex-Net Dataset
Codebase to train on the Dex-Net 3.0 suction grasp dataset. Model takes 32x32 depth image as input and outputs predicted grasp quality.
## Original Work
[Dex-Net Project Website](https://berkeleyautomation.github.io/dex-net/)

[Dex-Net Documentation](https://berkeleyautomation.github.io/dex-net/code.html)

[Dex-Net Package GitHub](https://github.com/BerkeleyAutomation/dex-net)

## Project Setup (Conda)
```
git clone https://github.com/BerkeleyAutomation/dexnew.git
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
View PyTorch's [Getting Started Page](https://pytorch.org/get-started/locally/) for other insallation options
Optionally for [Weights and Biases](https://wandb.ai/site) logging:
```
pip install wandb
```
