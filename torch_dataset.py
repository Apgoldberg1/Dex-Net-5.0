import torch
from torch.utils.data import Dataset
import numpy as np
import time
from torchvision import transforms


dataset_path = "dataset/dexnet_3/dexnet_09_13_17"

class Dex3Dataset(Dataset):
    def __init__(self, dataset_path, preload=True, num_files=2759):
        """
        dataset_path: uncompressed directory containing tensors folder
        preload: np.load all files upfront for faster performance
        num_files: How many files of the dataset to include, 2759 is number of files in DexNet3.0
        """
        self.num_files = num_files

        self.dataset_path = dataset_path
        self.dataset_len = 1000 * (self.num_files)        #1000 per file, 2760 is not a full 1000, numbering starts at 0
        self.preload = preload

        self.transformations = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Random horizontal flipping
            transforms.RandomVerticalFlip(),    # Random vertical flipping
            transforms.RandomRotation(180),     # Random 180-degree rotation
        ])

        def preload(var):
            """
            var: camera_intrs, camera_poses, collistion_free, depths_ims_tf_table, grasp_ids, hand_poses, image_labels, obj_labels, obj_masks, pose_labels, robust_suction_wrench_resistance
            """
            assert var in ["camera_intrs", "camera_poses", "collistion_free", "depth_ims_tf_table", "grasp_ids", "hand_poses", "image_labels", "obj_labels", "obj_masks", "pose_labels", "robust_suction_wrench_resistance"]

            return [
                np.load(dataset_path + "/tensors/" + var + "_" + str(file_num).zfill(5) + ".npz")["arr_0"]
                for file_num in range(self.num_files)
                ]


        if preload:
            start = time.time()
            print("STARTING PRELOAD")

            self.depth_im_data = preload("depth_ims_tf_table")
            self.grasp_metric_data = preload("robust_suction_wrench_resistance")
            self.hand_poses = preload("hand_poses")

            print("FINISIHED PRELOAD took:", time.time() - start)

    def __len__(self):
        return self.num_files * 1000
    
    def __getitem__(self, idx):
        file_num = str(idx // 1000).zfill(5)        #pad file_num with 0s to match file names
        data_idx_num = idx % 1000       #index within file

        if self.preload:
            depth_im = self.depth_im_data[idx // 1000][data_idx_num]
            grasp_metric = self.grasp_metric_data[idx // 1000][data_idx_num]
            hand_pose = self.hand_poses[idx // 1000][data_idx_num][2:4]       #we only want the third column which is z
        else:
            depth_im = np.load(dataset_path + "/tensors/depth_ims_tf_table_" + file_num + ".npz")["arr_0"][data_idx_num]
            grasp_metric = np.load(dataset_path + "/tensors/robust_suction_wrench_resistance_"+ file_num + ".npz")["arr_0"][data_idx_num]
            hand_pose = np.load(dataset_path + "/tensors/hand_pose_"+ file_num + ".npz")["arr_0"][data_idx_num][:,2:4]



        depth_im_t, hand_pose_t, grasp_metric_t = torch.tensor(depth_im), torch.tensor(hand_pose), torch.tensor(grasp_metric)
        depth_im_t = depth_im_t.reshape(1, 32, 32)
        depth_im_t = self.transformations(depth_im_t)

        depth_im_t = transforms.Resize((224, 224), antialias=True)(depth_im_t)
        

        #depth_im_t = (depth_im_t - depth_im_t.mean()) / depth_im_t.std()

        hand_pose_t = hand_pose_t.reshape(2)


        return depth_im_t, hand_pose_t, grasp_metric_t

def testLoader(num_files=100):
    from torch.utils.data import DataLoader
    dataset = Dex3Dataset(dataset_path, preload=True, num_files=num_files)

    batch_size = 128
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    print("full epoch benchmark on:", num_files * 1000, "datapoints")
    start = time.time()
    for i, batch in enumerate(dataloader):
        continue

    end = time.time() - start
    print("time:", end)

if __name__ == "__main__":
    testLoader(10)
    #var = "depth_ims_tf_table"
    #file_num = 15
    #out = np.load(dataset_path + "/tensors/" + var + "_" + str(file_num).zfill(5) + ".npz")["arr_0"]
    #print(out)
    #print(out.mean())
    #print(out.shape)

"""
dataset has:
camera_intrs
camera_poses
collistion_free
depths_ims_tf_table
grasp_ids
hand_poses
image_labels
obj_labels
obj_masks
pose_labels
robust_suction_wrench_resistance
"""
