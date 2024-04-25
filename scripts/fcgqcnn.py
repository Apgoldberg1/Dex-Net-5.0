"""
Run inference with FCGQCNN
"""
import torch
import numpy as np
import math
from dexnet.grasp_model import fakeSuctionFCGQCNN
from dexnet.grasp_model import HighRes as FCGQCNN
from dexnet.grasp_model import DexNetBase as GQCNN
import matplotlib.pyplot as plt
import cv2
import os

def processRGB(img_path: str):
    img = cv2.load(img_path)
    img = np.mean(img, axis=-1)
    return processNumpy(img)

def processNumpy(img):
    """
    example preprocessing for numpy depth image shape (width, height, 1) before inference
    blurs, normalizes, resizes, pads, and converts to tensor
    NOT a batched operation.
    """
    img = blur(img)
    img = (img - img.mean()) / img.std()
    img = cv2.resize(img, (40, 40))

    pad = 15
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    img = torch.tensor(img, dtype=torch.float32).squeeze().unsqueeze(0)

    return img


def blur(img):
    kernel = np.ones((15,15), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (15, 15), 5)

    return img

def main(fcgqcnn_weights_path, img, device):
    fcgqcnn = FCGQCNN()
    fcgqcnn.load_state_dict(torch.load(fcgqcnn_weights_path))

    x_show = img
    x = processNumpy(img)
    x = torch.unsqueeze(0)

    assert len(x.shape) == 4, f"shape should be (batch, 1, x, y), but shape is {x.shape}"
    x = x.to(device)

    fcgqcnn.eval()
    fcgqcnn.to(device)
    with torch.no_grad():
        output = fcgqcnn(x)[0]

    output = output.cpu()
    print("output shape", output.shape)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    torch.save(output.numpy(), "outputs/fcgqcnn.npy")

    fig, axes = plt.subplots(1, 2, figsize=(10, 8))

    axes[0].imshow(output.numpy().squeeze(), cmap="gray")
    axes[0].axis('off')

    axes[1].imshow(x_show.squeeze(), cmap="gray")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("outputs/out.png")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process configuration file.")
    parser.add_argument(
        "--model_path",
        dest="model_path",
        metavar="MODEL_FILE_PATH",
        type=str,
        required=True,
        help="Path to FC-GQ-CNN model checkpoint",
    )
    parser.add_argument(
        "--img_path",
        dest="img_path",
        metavar="IMG_FILE_PATH",
        type=str,
        required=True,
        help="Grey-scale depth image JPG",
    )

    args = parser.parse_args()
    fcgqcnn_weights_path = args.model_path
    img = cv2.imread(args.img_path)
    img = np.mean(img, axis=-1).unsqueeze(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: running inference on cpu")

    main(fcgqcnn_weights_path, img, device)
