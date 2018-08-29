import cv2
import random
import math
import numpy as np
from config import CONFIG

"""
1. crop to center
2. rotation
3. flip
h5文件已经对数据做了处理，每个样本是单个人
"""


# [0.7,1.3)
def resize(orgImgs, centers, gt_joints_list):       
    output_imgs = []
    output_centers = []
    output_joints_list = []
    for i in range(len(orgImgs)):  
        img = cv2.imread(CONFIG.image_dir + orgImgs[i].decode('utf-8'))  
        rdn = random.random()       
        rdn = (1.3-0.7)*rdn + 0.7
        img = cv2.resize(img, (0,0), fx=rdn, fy=rdn, interpolation=cv2.INTER_LINEAR)  
        output_imgs.append(img)
        center_x = float(np.float64(centers[i][0])) * rdn
        center_y = float(np.float64(centers[i][1])) * rdn
        output_centers.append(np.array([center_x, center_y]))      
        output_joints_list.append(np.array(gt_joints_list[i])*rdn)      
    return output_imgs, output_centers, output_joints_list


#以人体中心crop图片 targetSize*targetSize 大小
def crop_to_center(orgImgs, targetSize, scales, centers, gt_joints_list, is_train):
    output_imgs = []
    output_joints_list = []
    for i in range(len(orgImgs)):       
        crop_img = np.ones((368, 368, 3)) * 128     
        img = orgImgs[i]
        if not is_train:        
            img = cv2.imread(CONFIG.image_dir + img.decode( 'utf-8')) 

        s = 200 * scales[i] / targetSize  
        img_resized = cv2.resize(img, (0, 0), fx=1/s, fy=1/s, interpolation=cv2.INTER_LINEAR)  
        center_x = float(np.float64(centers[i][0]))/s
        center_y = float(np.float64(centers[i][1]))/s
        half = targetSize / 2  

        # 图像中的截取的部分
        x0 = max(0, center_x - half)        
        y0 = max(0, center_y - half)
        x1 = min(img_resized.shape[1], center_x + half)
        y1 = min(img_resized.shape[0], center_y + half)

        # 目标中需要填充的部分
        fromx0 = max(0, half - center_x)
        fromx1 = (img_resized.shape[1] - center_x + half) if (img_resized.shape[1] - center_x) < half else 2 * half
        fromy0 = max(0, half - center_y)
        fromy1 = (half + img_resized.shape[0] - center_y) if (center_y + half) > img_resized.shape[0] else 2 * half

        crop_img[int(fromy0 + 0.5):int(fromy1 + 0.5), int(fromx0 + 0.5):int(fromx1 + 0.5), :] = \
            img_resized[int(y0 + 0.5):int(y1 + 0.5), int(x0 + 0.5):int(x1 + 0.5), :]
        output_imgs.append(crop_img)

        #调整ground truth的关节点坐标
        joints = gt_joints_list[i]          
        joints = np.array(joints)*1/s       
        output_joints = []
        for coord in joints:
            if coord[0]<=0.0 and coord[1]<=0.0:     
                output_joints.append([-1000,-1000])
            else:
                output_joints.append([coord[0]+half-center_x, coord[1]+half-center_y])  
        output_joints_list.append(output_joints)
    return output_imgs, output_joints_list

#图片旋转
def rotate(imgs, joints_list):
    deg = random.uniform(-40.0, 40.0)       
    rotate_imgs = []
    rotate_joints_list = []
    for i in range(len(imgs)):      
        img = imgs[i]
        joints = joints_list[i]
        rot_matrix = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), deg, 1)  
        img = cv2.warpAffine(img, rot_matrix, img.shape[1::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=128)
        
        adjust_joint = []
        for coord in joints:
            r_coord = _rotate_coord(rot_matrix, coord)
            adjust_joint.append(r_coord)
        rotate_imgs.append(img)
        rotate_joints_list.append(adjust_joint)
    return rotate_imgs, rotate_joints_list

def _rotate_coord(matrix, point):      
    px, py = point      
    return np.matmul(matrix, [px, py, 1])
    # angle = -1 * angle / 180.0 * math.pi
    # ox, oy = shape
    # px, py = point      
    # ox /= 2         
    # oy /= 2         
    # qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)      
    # qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    # return int(qx + 0.5), int(qy + 0.5)


#水平翻转图片
def flip(imgs, joints_list):
    r = random.uniform(0, 1.0)
    if r > 0.5:
        return imgs, joints_list    
    flip_imgs = []
    flip_joints_list = []
    for i in range(len(imgs)):      
        img = imgs[i]
        joints = joints_list[i]
        img = cv2.flip(img, 1)      
        flip_imgs.append(img)
        adjust_joint = []
        for coord in joints:
            adjust_joint.append((img.shape[1] - coord[0], coord[1]))  
        flip_joints_list.append(adjust_joint)
    return flip_imgs, flip_joints_list




