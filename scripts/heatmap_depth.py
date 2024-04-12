import torch
from dexnet.grasp_model import fakeSuctionFCGQCNN as FCGQCNN, DexNet3
import glob
import cv2
import os
import numpy as np

dex = DexNet3()
dex.load_state_dict(torch.load("dexnew/model_zoo/best_replica.pth"))
model = FCGQCNN(dex)
model.eval()

depth_dir = "mesh_data_dir/ns_data/CokePlasticSmallGrasp_800_tex/depth"
#depth_dir = "main_mesh_data/mini_mesh_set_pj/ns_data/1a04e3eab45ca15dd86060f189eb133/depth"

num_imgs = 100
resize_dim = 50
model_input = torch.zeros((num_imgs, 1, resize_dim, resize_dim))

z = torch.zeros((num_imgs, 2))
z[:, 0] = (.6 - .566) / .06
z[:, 1] = (1.65 - .36) / .23

for i in range(num_imgs):
    frame = np.load(os.path.join(depth_dir, f"np_{i}.npy"))
    frame = (frame - frame.mean()) / frame.std()
    frame = cv2.resize(frame, (resize_dim, resize_dim))
    frame = frame
    model_input[i] = torch.from_numpy(frame).unsqueeze(0)

model, model_input, z = model.to("cuda"), model_input.to("cuda"), z.to("cuda")

with torch.no_grad():
    outputs = model(model_input, z).to("cpu")

np.save(os.path.join(depth_dir, "..", "heatmaps.npy"), outputs.numpy())

heatmap_dir = os.path.join(depth_dir, "..", "heatmaps")
os.makedirs(heatmap_dir, exist_ok=True)

for i in range(num_imgs):
    np.save(os.path.join(heatmap_dir, f"np_{i}.npy"), outputs[i].squeeze().numpy())

import matplotlib.pyplot as plt
for i in range(10):
    #imgs = model_input.cpu().numpy()[i*8].squeeze()
    #to_show = np.stack([(outputs.numpy()[i*8]).squeeze() * 255, imgs, np.zeros_like(imgs)]).transpose([1, 2, 0])
    #plt.imshow(to_show)
    #plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(outputs.numpy()[i*9].squeeze(), cmap="gray")
    
    axes[0].axis('off')

    axes[1].imshow(np.load(os.path.join(depth_dir, f"np_{9*i}.npy")).squeeze(), cmap="gray")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


#for i, file_path in enumerate(glob.glob(os.path.join(depth_dir, "depth/*.npy")):
    #frame = np.load(file_path)
    
