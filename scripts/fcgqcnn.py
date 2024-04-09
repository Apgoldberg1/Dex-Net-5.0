import torch
import numpy as np
import math
from grasp_model import fakeSuctionFCGQCNN as FCGQCNN
from grasp_model import DexNet3
import matplotlib.pyplot as plt
import cv2

depth_im = "test_data/depth/np_10.npy"
#depth_im = "test_data/demo_depth.npy"
#depth_im = "test_data/dexnet_depth.npy"
x = np.load(depth_im)
#x = x["arr_0"][0]

model_weights = "test_data/model_w/best_replica.pth"

gqcnn = DexNet3()
gqcnn.load_state_dict(torch.load(model_weights))

fcgqcnn = FCGQCNN(gqcnn).to("cuda")


def norm_clip_depth(depth_img):
    depth_cliped = np.clip(depth_img, 0.6, 0.9)
    depth_cliped = np.exp(depth_cliped)
    depth_cliped_normalized = (depth_cliped - depth_cliped.min())/(depth_cliped.max() - depth_cliped.min())
    return (1 - depth_cliped_normalized)

def clip_background(depth_img):
    depth_img[depth_img > 5] = 1
    return depth_img

def blur(img):
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Example: 5x5 circular kernel
    kernel = np.ones((15,15), np.uint8)

    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (15, 15), 5)
    return img


normalizers = (0.59784445, 0.00770147890625, 0.5667523, 0.06042659375, 0.360944025, 0.231009775)       #mean, std (x3) image, pose dist, pose angle
def norm_data(img, pose):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # Example: 5x5 circular kernel
    #kernel = np.ones((20,20), np.uint8)

    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img, (15, 15), 5)
    img = cv2.resize(img, (50, 50))
    #img = cv2.resize(img, (195, 263))
    #img = img[50:150, 50:200]

    img = (img - normalizers[0]) / normalizers[1]

    pose = pose.reshape(2)

    pose[0], pose[1] = (pose[0] - math.copysign(1, pose[0]) * normalizers[2]) / normalizers[3], (pose[1] - math.copysign(1, pose[1]) * normalizers[4]) / normalizers[5]

    return img, pose

z = torch.zeros((1, 2))
z[:, 0] = .6
z[:, 1] = 1.65

#x = norm_clip_depth(x)
print(x.min())
#x = clip_background(x)
x = blur(x)
#x_show = norm_clip_depth(x) + .2
#x = a=np.abs(x.max() - x)
x_show = x
_, z = norm_data(x_show, z)
x = (x - x.mean()) / x.std()
x = cv2.resize(x, (40, 40))
x_show = x
x = x.squeeze()

x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
x, z = x.to("cuda"), z.to("cuda")
z = z.unsqueeze(0)

fcgqcnn.eval()
with torch.no_grad():
    output = fcgqcnn(x, z)

print(torch.max(output))
output = output.to("cpu")
torch.save(output.numpy(), "test_data/outputs/out.npy")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(output.numpy().squeeze(), cmap="gray")
axes[0].axis('off')

axes[1].imshow(x_show.squeeze(), cmap="gray")
axes[1].axis('off')

plt.tight_layout()
plt.show()

#plt.imshow(output.numpy().squeeze(), cmap="gray")
#plt.show()