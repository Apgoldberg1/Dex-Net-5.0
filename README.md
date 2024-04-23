# Dex-Net 5.0 - A PyTorch implementation to train on the Dex-Net Dataset
Dex-Net 5.0 is a PyTorch implementation to train on the Dex-Net 2.0 parallel jaw grasp and Dex-Net 3.0 suction grasp dataset. It provides faster data-loading, training, and inference over the original implementations. The models take cropped 32x32 depth images, the distance of the gripper from the camera, and grasp approach angle (only for suction) as input, and output a grasp confidence.

## Original Work
Dex-Net 5.0 is an extension of previous work which can be found here:

[Dex-Net Project Website](https://berkeleyautomation.github.io/dex-net/)

[Dex-Net Documentation](https://berkeleyautomation.github.io/dex-net/code.html)

[Dex-Net Package GitHub](https://github.com/BerkeleyAutomation/dex-net)

## Project Setup

View PyTorch's [Getting Started Page](https://pytorch.org/get-started/locally/) for PyTorch installation options

```
git clone https://github.com/BerkeleyAutomation/dexnew.git
cd Dex-Net-5.0
pip install -e .
```

The Dex-Net dataset used for this repository can be downloaded [here]()

Other published datasets and mesh files can be found [here](https://drive.google.com/drive/u/1/folders/1-6o1-AlZs-1WWLreMa1mbWnXoeIEi14t)

"DexNetGQCNN" suction grasp model weights can be dowloaded [here]()

"EfficientNet" suction grasp model weights can be downloaded [here]()

"DexNetFCGQCNN" suction grasp model weights can be downloaded [here]()

"DexNetGQCNN" parallel jaw grasp model weights can be downloaded [here])()

"DexNetFCGQCNN" parallel jaw grasp model weights can be downloaded [here]()

See grasp models section for more model details.

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

Defines models. Models take in a batch of normalized single channel depth images and output grasp confidence(s). See grasp models section for more details.

### torch_dataset.py

Provides PyTorch dataset to load the Dex-Net 3.0 and Dex-Net 2.0 dataset.

## grasp models

"DexNetGQCNN" is an implementation similar to the model described in [Dex-Net 2.0](https://arxiv.org/pdf/1703.09312.pdf). Unlike the original implementation, it doesn't take the gripper z distance as input because this was not found to impact training. It takes only the 32x32 normalized depth images.

"EfficientNet" uses PyTorch's efficientnet_b0 implementation with an additional linear layer and softmax. It slightly out performs "DexNetGQCNN" (see results).

"DexNetFCGQCNN" is a fully convolutional network which takes a batch of normalized depth images which may be larger than 32x32 and returns a grasp confidence heatmap. DexNetGQCNN weights can be converted to DexNetFCGQCNN weights using convert_weights.py.

"fakeFCGQCNN" runs a provided GQCNN across each 32x32 crop of an image to return a grasp confidence heatmap. This model is inefficient and is intended for testing and benchmarking purposes.


