# Dex-Net 5.0 - A PyTorch implementation to train on the Dex-Net Dataset
Dex-Net 5.0 is a PyTorch implementation to train on the Dex-Net 2.0 parallel jaw grasp and Dex-Net 3.0 suction grasp dataset. It provides faster data-loading, training, and inference over the original implementations. The models take cropped 32x32 depth images, the distance of the gripper from the camera, and grasp approach angle (only for suction) as input, and output a grasp confidence.
## Original Work
Dex-Net 5.0 is an extension of previous work which can be found here:

[Dex-Net Project Website](https://berkeleyautomation.github.io/dex-net/)

[Dex-Net Documentation](https://berkeleyautomation.github.io/dex-net/code.html)

[Dex-Net Package GitHub](https://github.com/BerkeleyAutomation/dex-net)

## Project Setup
```
git clone https://github.com/BerkeleyAutomation/dexnew.git
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
View PyTorch's [Getting Started Page](https://pytorch.org/get-started/locally/) for other insallation options

Optionally for [Weights and Biases](https://wandb.ai/site) logging
```
pip install wandb
```
Set wandb to false in the YAML config to disable logging.

The dataset can be downloaded [here](https://drive.google.com/drive/u/1/folders/1-6o1-AlZs-1WWLreMa1mbWnXoeIEi14t)

Dex-Net 3.0 suction grasp model weights can be dowloaded [here](https://drive.google.com/file/d/1dSHhD0lbySvPZGPN8XJbh9aCcJcWVJ1s/view?usp=sharing)

## Usage

### train_model.py

Runs train eval loop based on the given config.

```
python3 train_model.py --config PATH_TO_DESIRED_CONFIG_FILE
```

### analyze.py

For model benchmarks. Generates precision recall curve, inference speed timings, and mean and std over the datset.

### grasp_model.py

Defines models to be used in train_model.py

"DexNet3" is an implementation of the model described in [Dex-Net 3.0](https://arxiv.org/abs/1709.06670) and provides the best performance.

### torch_dataset.py

Provides PyTorch dataset to load the Dex-Net 3.0 dataset efficiently
```
python3 analyze.py --model_file PATH_TO_MODEL_WEIGHTS --model_name dexnet3
```



