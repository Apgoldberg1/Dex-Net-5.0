"""
Run inference with FCGQCNN
"""
import torch
import numpy as np
import math
from dexnet.grasp_model import fakeSuctionFCGQCNN
#from dexnet.grasp_model import DexNet3FCGQCNN as FCGQCNN
from dexnet.grasp_model import HighRes as FCGQCNN
from dexnet.grasp_model import DexNet3
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
    NOT batched operation.
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

def main(fcgqcnn_weights_path, dexnet3_weights_path, img):
    gqcnn = DexNet3()
    gqcnn.load_state_dict(torch.load(dexnet3_weights_path))
    fake_model = fakeSuctionFCGQCNN(gqcnn)

    fcgqcnn = FCGQCNN()
    fcgqcnn.load_state_dict(torch.load(fcgqcnn_weights_path))

    x_show = img
    print(img.shape)
    x = processNumpy(img)
    x = torch.stack([x,x,x]) # add batch dim
    print(x.shape)
    assert len(x.shape) == 4, f"shape should be (batch, 1, x, y), but shape is {x.shape}"
    x = x.to("cuda")

    fcgqcnn.eval()
    fcgqcnn.to("cuda")
    fake_model.eval()
    fake_model.to("cuda")
    with torch.no_grad():
        output = fcgqcnn(x)[0]
        output_fake = fake_model(x)[0]

    output = output.to("cpu")
    print("output shape", output.shape)
    output_fake = output_fake.to("cpu")

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    torch.save(output.numpy(), "outputs/fcgqcnn.npy")
    torch.save(output_fake.numpy(), "outputs/fake_fcgqcnn.npy")

    fig, axes = plt.subplots(1, 3, figsize=(10, 8))

    axes[0].imshow(output.numpy().squeeze(), cmap="gray")
    axes[0].axis('off')

    axes[1].imshow(output_fake.numpy().squeeze(), cmap="gray")
    axes[1].axis('off')

    axes[2].imshow(x_show.squeeze(), cmap="gray")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("outputs/out.png")
    plt.show()
    

if __name__ == "__main__":
    fcgqcnn_weights_path = "model_zoo/Dex-Net-3-fcgqcnn.pt"
    dexnet3_weights_path = "model_zoo/Dex-Net-3-gqcnn.pth"
    img = np.load("mesh_data_dir/depth/np_0.npy")
    main(fcgqcnn_weights_path, dexnet3_weights_path, img)