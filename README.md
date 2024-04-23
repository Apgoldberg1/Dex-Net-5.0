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
python3 scripts/train_model.py --config PATH_TO_DESIRED_CONFIG_FILE
```

### analyze.py

Contains functions to generate precision recall curves, run inference speed timings, and compute the mean and std over the datset.
```
python3 scripts/analyze.py --model_path PATH_TO_MODEL --model_name [DexNetGQCNN, EfficientNet]
```

### grasp_model.py

Defines models to be used in train_model.py. Models take in a batch of normalized single channel depth images and output grasp confidence(s).

"DexNetGQCNN" is an implementation similar to the model described in [Dex-Net 2.0](https://arxiv.org/pdf/1703.09312.pdf). Unlike the original implementation, it doesn't take the gripper z distance as input because this was not found to impact training. It takes only the 32x32 normalized depth images.
"EfficientNet" uses PyTorch's efficientnet_b0 implementation with an additional linear layer and softmax. It slightly out performs "DexNetGQCNN" (see results).
"DexNetFCGQCNN" is a fully convolutional network which takes a batch of normalized depth images which may be larger than 32x32 and returns a grasp confidence heatmap. DexNetGQCNN weights can be converted to DexNetFCGQCNN weights using convert_weights.py.
"fakeFCGQCNN" runs a provided GQCNN across each 32x32 crop of an image to return a grasp confidence heatmap. This model is inefficient and is intended for testing and benchmarking purposes.


### torch_dataset.py

Provides PyTorch dataset to load the Dex-Net 3.0 and Dex-Net 2.0 dataset.



