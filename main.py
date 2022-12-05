

import os
import glob
from tqdm import tqdm
from utils.kitti_utils import *
from utils.calibration import *
if __name__ == "__main__":

    root = "data/object/training"

    lidar_data = sorted(glob.glob(os.path.join(root,"velodyne","*")))
    img_data = sorted(glob.glob(os.path.join(root,"image_2","*")))
    calib = sorted(glob.glob(os.path.join(root,"calib","*")))

    output_dir = "result/"
    os.makedirs(output_dir,exist_ok=True)

    for idx in tqdm(range(len(lidar_data)),mininterval=1):
        # points = load_velo_points(lidar_data[idx])
        # Calibration(calib[idx])
        generate_depth_map(
            lidar_path=lidar_data[idx], 
            rgb_path=img_data[idx], 
            calib_path = calib[idx]
        )
