import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time,os,json,random,cv2
from config import CONFIG
from collections import defaultdict
import data_process
import h5py
from config import CONFIG
"""
0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 
5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 
10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
"""

def assemble_data(data_file, batch_size):
    print('loading annotations into memory...')
    data = h5py.File(data_file, 'r')  
    total_size = len(data['center'])        

    for i in range(total_size):
        idxs = np.random.randint(0, total_size, batch_size)      
        output_imgs, centers, output_joints_list = data_process.resize(np.array(data['imgname'])[idxs], np.array(data['center'])[idxs],
                                                                       np.array(data['part'])[idxs])
        ctc_imgs, ctc_gt_joints = data_process.crop_to_center(output_imgs, CONFIG.input_size, np.array(data['scale'])[idxs],
                                                                  centers, output_joints_list, True)            
        rotate_imgs, rotate_gt_joints = data_process.rotate(ctc_imgs, ctc_gt_joints)
        flip_imgs, flip_gt_joints = data_process.flip(rotate_imgs, rotate_gt_joints)

        yield np.array(flip_imgs), np.array(flip_gt_joints)


