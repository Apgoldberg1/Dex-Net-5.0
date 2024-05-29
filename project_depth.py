import numpy as np
import cv2
import glob
import os
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from autolab_core import CameraIntrinsics, DepthImage, RigidTransform
import matplotlib
import viser
import time

data_path = "/home/apgoldberg/Dex-Net-5.0/gogs_test_data_real/test"

def load_zed_depth_folder(data_path, img_shape=(720, 1280)):
    num_files = len(os.listdir(os.path.join(data_path, "depth")))
    depth_ims = np.zeros((num_files, img_shape[0], img_shape[1], 1))
    for i in range(num_files):
        float_values = np.load(os.path.join(data_path, "depth", f"{i}.npy"))
        depth_ims[i] = float_values.reshape(img_shape[0], img_shape[1], 1).astype(np.float32)
    return depth_ims

def load_poses_folder(data_path):
    num_files = len(os.listdir(os.path.join(data_path, "depth")))
    poses = np.zeros((num_files, 4, 4))
    for i in range(num_files):
        float_values = np.loadtxt(os.path.join(data_path, "cam_poses", f"{i}.txt"))
        poses[i] = float_values
    return poses

#depth_arr = load_zed_depth_folder(data_path, img_shape=(720, 1280))[::5]
depth_arr = load_zed_depth_folder(data_path, img_shape=(720, 1280))[20:30]
poses = load_poses_folder(data_path)[20:30]

print("data loaded")

intrinsics = CameraIntrinsics(
    frame = "base",
    fx = 733.1768188476562,
    fy = 733.1768188476562, 
    cx = 658.0925903320312, 
    cy = 352.3837890625
)

# deproject_arr = np.array([[0, 0, 0]])
deproj_arr_list = []
for i, depth_im in enumerate(depth_arr):
    depth_im_core = DepthImage(depth_im, "base")
    deprojection = poses[i][:3,:3] @ intrinsics.deproject(depth_im_core).subsample(10)[0].data + poses[i][:3, 3].reshape((3, 1))
    # deprojection = RigidTransform(poses[i][:3,:3], poses[i][:3, 3]).apply(intrinsics.deproject(depth_im_core))
    deprojection_data = deprojection.transpose()
    # deproject_arr = np.concatenate((deproject_arr, deprojection_data))
    deproj_arr_list.append(deprojection_data)

# deprojection_data = deproject_arr


server = viser.ViserServer()

viridis = matplotlib.colormaps['viridis']
# normalized_depth = 5 * (deproject_arr[:,2] - deprojection_data[:,2].mean()) / deprojection_data[:,2].ptp()
# colors = viridis(normalized_depth.transpose())[:,:3]
colors=[0,0,0]

# deprojection_data[:,2] = -deprojection_data[:,2]
# server.add_point_cloud("cloud", deprojection_data, colors=colors, position=(0,0,0), point_size=.001, point_shape='circle')
for idx, arr in enumerate(deproj_arr_list):
    server.add_point_cloud(f"cloud_{idx}", arr, colors=colors, position=(0,0,0), point_size=.005, point_shape='circle')

while True:
    time.sleep(10.0)

# trimesh_pointcloud.show()
