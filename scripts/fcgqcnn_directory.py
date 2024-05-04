"""
Go through a directory of blender-rendered depth images and output a grasp map for each of them.
"""
import torch
import numpy as np
import math
from dexnet.grasp_model import DexNet3FCGQCNN as FCGQCNN
from dexnet.grasp_model import DexNet3, fakeSuctionFCGQCNN
import matplotlib.pyplot as plt
import cv2
import os

fake_fc = False # set to True for for-loop, False for FCGQCNN beta

# change directory as desired (should probably be a CLI arg)
#directory = '/home/ryanhoque/scene_data2/CokePlasticSmallGrasp_800_tex'
directory = '/home/ryanhoque/scene_data/more_objects/book'
if not os.path.exists(directory+'/grasp'):
    os.makedirs(directory+'/grasp')
    os.makedirs(directory+'/grasp_viz')

# load model
model_weights_path = "model_zoo/fcgqcnn_conversion.pt"
fcgqcnn = FCGQCNN()
fcgqcnn.load_state_dict(torch.load(model_weights_path))
fcgqcnn.eval()
fcgqcnn.to("cuda")

dexnet3_weights_path = "model_zoo/just_image.pth"
gqcnn = DexNet3()
gqcnn.load_state_dict(torch.load(dexnet3_weights_path))
fake_model = fakeSuctionFCGQCNN(gqcnn)
fake_model.eval()
fake_model.to('cuda')

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

for i in range(100):
    #img = cv2.imread("{}/depth/img_{}.png".format(directory, i)).mean(axis=2)
    img = cv2.imread("{}/depth/{}.png".format(directory, str(i).zfill(4))).mean(axis=2)
    img = process_img(img)
    output_path = "{}/grasp/{}.npy".format(directory, i)
    with torch.no_grad():
        if fake_fc:
            output = fake_model(img)
        else:
            output = fcgqcnn(img)
        output = output.to('cpu').numpy().squeeze()
        print(output.min(), output.max())
        # threshold value
        output[output < 0.2] = 0.
        output = cv2.resize(output, (512,512))
    np.save(output_path, output)
    cv2.imwrite("{}/grasp_viz/{}.png".format(directory, i), (output * 255).astype(np.uint8))