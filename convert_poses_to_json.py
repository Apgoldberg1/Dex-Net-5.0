poses_dir = "gogs_test_data_real/manual_fixed_intr/poses"
img_dir = "img"

import glob
import os
import numpy as np
import json
import math
from autolab_core import RigidTransform

def get_rot_matrix(angle, axis):
    cos, sin = math.cos, math.sin
    if axis == 'Z':
        rot_matrix = np.array([
            [cos(angle), -sin(angle), 0],
            [sin(angle), cos(angle), 0],
            [0, 0, 1]
            ])
    elif axis == 'Y':
        rot_matrix = np.array([
            [cos(angle), 0, sin(angle)],
            [0, 1, 0],
            [-sin(angle), 0, -cos(angle)]
            ])
    elif axis == 'X':
        rot_matrix = np.array([
            [1, 0, 0],
            [0, cos(angle), -sin(angle)],
            [0, sin(angle), cos(angle)]
            ])
    else:
        raise AssertionError("axis must be 'X', 'Y', or 'Z'")

    return rot_matrix

def save_json(data, filename):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

extrinsics_dicts = []
print(os.listdir(poses_dir))
num_files = len(os.listdir(poses_dir))
print(num_files)

cam_to_wrist = RigidTransform.load("T_webcam_wrist.tf")
cam_to_wrist._from_frame = "world"
cam_to_wrist._to_frame = "world"
for i in range(num_files):
    transform_mat = np.loadtxt(os.path.join(poses_dir, f"{i}.txt"))
    #transform_mat = transform_mat @ cam_to_wrist.inverse().matrix
    # transform_mat = RigidTransform(transform_mat[:3,:3], transform_mat[:3, 3], from_frame="world").matrix @ cam_to_wrist.inverse().matrix @ cam_to_wrist.inverse().matrix
    transform_mat[:3,:3] = transform_mat[:3,:3] @ get_rot_matrix(np.pi, 'X')
    #transform_mat.rotation = transform_mat.rotation @ get_rot_matrix(np.pi, 'X')
    extrinsics_dicts.append({
        "file_path": os.path.join(img_dir, f"{i}.jpg"),
        "transform_matrix": transform_mat.tolist()
        # "transform_matrix": transform_mat.matrix.tolist()
        })
    
intrinsics_dict = {
    "w": 1280,
    "h": 720,
    "fl_x": 733.1768188476562,
    "fl_y": 733.1768188476562, 
    "cx": 658.0925903320312, 
    "cy": 352.3837890625,
    # "fl_x": 734.7650862680024,
    # "fl_y": 734.5320186866934,
    # "cx": 625.8374429422679,
    # "cy": 365.6027800515266,
    "k1": 0.0015874442384528349,
    "k2": -0.0004268132666342962,
    "p1": -0.00047194861724516224,
    "p2": 0.0012418137485269722,
    "camera_model": "OPENCV",
    }
        
intrinsics_dict["frames"] = extrinsics_dicts
save_json(intrinsics_dict, os.path.join(poses_dir, "..", "transforms.json"))
