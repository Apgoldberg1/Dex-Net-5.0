"""
Go through a directory of directories of blender-rendered depth images and output a grasp map for each of them.
"""
import torch
import numpy as np
import math
from dexnet.grasp_model import DexNet3FCGQCNN as FCGQCNN
from dexnet.grasp_model import DexNet3, fakeSuctionFCGQCNN
import matplotlib.pyplot as plt
import cv2
import os

def list_directories(parent_dir):
    # List all entries in the specified directory
    entries = os.listdir(parent_dir)
    # Filter out only directories
    directories = [entry for entry in entries if os.path.isdir(os.path.join(parent_dir, entry))]
    return directories

# load model
model_weights_path = "model_zoo/fcgqcnn_conversion.pt"
fcgqcnn = FCGQCNN()
fcgqcnn.load_state_dict(torch.load(model_weights_path))
fcgqcnn.eval()
fcgqcnn.to("cuda")

def process_img(x):
    x = 255 - x # blender depth is inverted
    # blur
    kernel = np.ones((15,15), np.uint8)
    x = cv2.dilate(x, kernel, iterations=1)
    x = cv2.GaussianBlur(x, (15, 15), 5)
    x = (x - x.mean()) / x.std()
    x = cv2.resize(x, (40, 40))
    pad = 15
    x = cv2.copyMakeBorder(x, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    x = x.squeeze()
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to("cuda")
    return x

parent_directory = '/home/ryanhoque/scene_data/3dnet'
directories = list_directories(parent_directory)

j = 0
for directory in directories:
    print('directory', directory, j)
    path = os.path.join(parent_directory, directory)
    #if not os.path.exists(path+'/grasp_viz'):
    #    os.makedirs(path+'/grasp_viz')
    
    for i in range(100):
        img = cv2.imread("{}/depth/{}.png".format(path, str(i).zfill(4))).mean(axis=2)
        img = process_img(img)
        output_path = "{}/images/{}.npy".format(path, i)
        with torch.no_grad():
            output = fcgqcnn(img)
            output = output.to('cpu').numpy().squeeze()
            #print(output.min(), output.max())
            # threshold value
            output[output < 0.2] = 0.
            output = cv2.resize(output, (512,512))
        np.save(output_path, output)
        #cv2.imwrite("{}/grasp_viz/{}.png".format(path, i), (output * 255).astype(np.uint8))
    j = j + 1