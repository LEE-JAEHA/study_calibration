import numpy as np
import cv2
from utils.calibration import *
def load_velo_points(pcd_path):
    points = np.fromfile(pcd_path,dtype=np.float32)
    if points.shape[0] %4 ==0:
        points = points.reshape(-1,4)
        points[:3]=1 # make intensity all 1
    else:
        points = points.reshpae(-1,3)
        points = np.hstack((points,np.ones((points.shape[0],1))))
    return points


def generate_depth_map(lidar_path,rgb_path,calib_path):
    velo = load_velo_points(lidar_path) # (nx4) homogeneous type
    img = cv2.imread(rgb_path) # 
    calib = Calibration(calib_path)

    P2 = calib.P # (3x4)
    # 1. velo -> homogeneous type : already done
    #    output : nx4
    # 2. velo to reference : need [velo to cam] 
    #    output : (nx3)

    velo2ref = np.dot(calib.V2C,velo.T) # (3x4) * (4xn) => 3xn
    velo2ref = velo2ref.T # (nx3)

    # 3. reference to rect : need R0
    #    output : nx3
    # np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
    ref2rect = np.dot(calib.R0,velo2ref.T) # (3x3) * (3xn) => 3xn
    ref2rect =ref2rect.T # (nx3)

    # 4. rect to img
    #    output : nx3
    n = ref2rect.shape[0]
    rect2img = np.hstack((velo2ref,np.ones((n,1)))) # to homogeneous -> (nx4)
    velo2img = np.dot(rect2img,calib.P.T) # (nx4) * (4x3) => (nx3)
    velo2img[:,:2] = velo2img[:,:2]/velo2img[:,2][...,np.newaxis]
    

    val_inds = (velo2img[:,0]>=0) & (velo2img[:,1]>=0)
    val_inds = val_inds & (velo2img[:,0]<img.shape[1]) & (velo2img[:,1]<img.shape[0])
    val_inds = val_inds & (velo2img[:,2]>0)
    velo2img = velo2img[val_inds,:]

    depth = np.zeros(img.shape[:2])
    depth[velo2img[:,1].astype(np.int64),velo2img[:,0].astype(np.int64)]=velo2img[:,2]

    ### visual depth
    depth_trans = (depth-depth.min())/(depth.max()-depth.min())*255
    depth_trans[depth_trans>0]=depth_trans[depth_trans>0]+50
    depth_trans = depth_trans.astype("uint8")
    depth_trans = cv2.applyColorMap(depth_trans,cv2.COLORMAP_JET)
    depth_trans[depth==0,:] = np.array([0,0,0])

    output = cv2.resize(depth_trans,(img.shape[1],img.shape[0]))
    cv2.imshow("visual",output)
    while True:
        cv2.waitKey()
        k = cv2.waitKey(0)
        if k == 27:
            break
    cv2.destoryAllWindows()

    return depth
    