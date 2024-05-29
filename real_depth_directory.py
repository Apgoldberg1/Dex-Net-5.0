"""
Go through a directory of blender-rendered depth images and output a grasp map for each of them.
These are from real data so maybe treat them differently (nonsquare images, diff dimensions, don't blur as much, what else?)
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import os
import torch
from dexnet.grasp_model import HighResFCGQCNN as FCGQCNN
import glob
# from dexnet.grasp_model import DexNet3FCGQCNN as FCGQCNN

w = torch.load("/home/apgoldberg/Dex-Net-5.0/model_zoo/max_normal_normal_conversion.pth")
# w = torch.load("/home/apgoldberg/Dex-Net-5.0/model_zoo/ryans_fcgqcnn_conversion.pt")
model = FCGQCNN()
model.load_state_dict(w)
model.to("cuda")
model.eval()

def preprocess_img(x, normal_crop=None):
    kernel = np.ones((5,5), np.uint8)
    x = cv2.dilate(x, kernel, iterations=1)
    x = cv2.GaussianBlur(x, (5, 5), 5)
    if normal_crop is None:
        x = (x - x.mean()) / x.std()
    elif normal_crop == "normal":
        normalizers = (
            0.59784445,
            # 0.00770147890625,
            .06873399
        )  # mean, std value for depth images (from analyze.py script)
        x = (x - normalizers[0]) / normalizers[1]
    else:
        x_crop = x[normal_crop[0]:normal_crop[1],normal_crop[2]:normal_crop[3]]
        x = (x - x_crop.mean()) / x_crop.std()
    # pad = 15
    # x = cv2.copyMakeBorder(x, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    x = x.squeeze()
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to("cuda")
    return x

# change directory as desired (should probably be a CLI arg)
#directory = '/home/ryanhoque/scene_data2/CokePlasticSmallGrasp_800_tex'
directory = '/home/ryanhoque/scene_data/evonerf/multiglass'
if not os.path.exists(directory+'/grasp'):
    os.makedirs(directory+'/grasp')
    os.makedirs(directory+'/grasp_viz')

for i in range(51): # 89
    #img = cv2.imread("{}/depth/img_{}.png".format(directory, i)).mean(axis=2)
    img = cv2.imread("{}/depth/{}.png".format(directory, str(i).zfill(4))).mean(axis=2)
    img = process_img(img)
    output_path = "{}/grasp/img{}.npy".format(directory, i)
    with torch.no_grad():
        if fake_fc:
            output = fake_model(img)
        else:
            output = fcgqcnn(img)
        output = output.to('cpu').numpy().squeeze()
        print(output.min(), output.max())
        # threshold value
        output[output < 0.2] = 0.
        output[output > 1.] = 1.
        output = cv2.resize(output, (1280,720)) # 1600 x 896
    np.save(output_path, output)
    cv2.imwrite("{}/grasp_viz/{}.png".format(directory, i), (output * 255).astype(np.uint8))