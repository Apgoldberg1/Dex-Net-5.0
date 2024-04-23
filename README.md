# Dex-Net 5.0 - A PyTorch implementation to train on the Dex-Net Dataset
Dex-Net 5.0 is a PyTorch implementation to train on the Dex-Net 2.0 parallel jaw grasp and Dex-Net 3.0 suction grasp datasets. It provides faster data-loading, training, and inference over the original implementations. The models take normalized single channel depth images as input, and output grasp confidences.

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

The Dex-Net dataset used for this repository for training can be downloaded [here]()

The model weights for suction and parallel jaw grasp models can be found [here](). See grasp models section for more model details.

Other published datasets and mesh files from previous works can be found [here](https://drive.google.com/drive/u/1/folders/1-6o1-AlZs-1WWLreMa1mbWnXoeIEi14t)


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

Provides PyTorch dataset to load the Dex-Net 3.0 and Dex-Net 2.0 dataset. The dataset contains cropped 32x32 depth images of single objects paired with a grasp confidence score. For Dex-Net 2.0 data, this grasp metric corresponds to a parallel jaw grasp centered at the middle of the image and with the grasp axis horizontally on the image. For Dex-Net 3.0 suction data, this score corresponds to a suction grasp at the center of the image with the approach axis aligned to the middle column. See [Dex-Net 2.0](https://arxiv.org/abs/1703.09312) and [Dex-Net 3.0](https://arxiv.org/abs/1709.06670) for more details on dataset generation.

### configs

The configs include YAML files specifying model name, save name, dataset path, optimizer, wandb logging, batch size, and more. The dataset path should be to the directory containing the "tensors" folder for either the Dex-Net 2.0 or Dex-Net 3.0 datasets.

## grasp models

"DexNetGQCNN" is an implementation similar to the model described in [Dex-Net 2.0](https://arxiv.org/pdf/1703.09312.pdf). Unlike the original implementation, it doesn't take the gripper z distance as input because this was not found to impact training. It takes only the 32x32 normalized depth images.

"EfficientNet" uses PyTorch's efficientnet_b0 implementation with an additional linear layer and softmax. It slightly out performs "DexNetGQCNN" (see results).

"DexNetFCGQCNN" is a fully convolutional network which takes a batch of normalized depth images which may be larger than 32x32 and returns a grasp confidence heatmap. DexNetGQCNN weights can be converted to DexNetFCGQCNN weights using convert_weights.py.

"fakeFCGQCNN" runs a provided GQCNN across each 32x32 crop of an image to return a grasp confidence heatmap. This model is inefficient and is intended for testing and benchmarking purposes.


