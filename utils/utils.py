import numpy as np
import math
import cv2



# 生成高斯map
def generate_gaussian_map(img_height, img_width, c_x, c_y, sigma):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):           
            dist_sq = (x_p - c_x) * (x_p - c_x) + (y_p - c_y) * (y_p - c_y)         
            exponent = dist_sq / 2.0 / sigma / sigma            
            gaussian_map[y_p, x_p] = np.exp(-exponent)          
    return gaussian_map


def generate_gaussian_kernel(size, sigma=3, center=None):
    """
    :param size:  高斯map的大小
    :param sigma:  高斯核的标准差
    :param center:
    :return:
    """
    x = np.arange(0, size, 1, float)        
    y = x[:, np.newaxis]        

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]      
        y0 = center[1]     

    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / sigma / sigma)       


def generate_heatmaps_from_joints(input_size, heatmap_size, gaussian_variance, batch_gt_joints):
    """
    :param input_size: 图片尺寸
    :param heatmap_size: heatmap尺寸
    :param gaussian_variance: heatmap关节点高斯方差
    :param batch_gt_joints: groud truth 关节坐标
    :return:
    """
    scale_factor = input_size // heatmap_size       
    batch_gt_heatmaps = []
    for i in range(batch_gt_joints.shape[0]):      
        gt_heatmap = []
        bg_heatmap = np.ones(shape=(heatmap_size, heatmap_size))        
        for j in range(batch_gt_joints.shape[1]):       
            cur_joint_heatmap = generate_gaussian_kernel(heatmap_size, gaussian_variance, center=(batch_gt_joints[i][j] // scale_factor))
            gt_heatmap.append(cur_joint_heatmap)
            bg_heatmap -= cur_joint_heatmap     
        gt_heatmap.append(bg_heatmap)       
        batch_gt_heatmaps.append(gt_heatmap)
    batch_gt_heatmaps = np.asarray(batch_gt_heatmaps)
    batch_gt_heatmaps = np.transpose(batch_gt_heatmaps, (0, 2, 3, 1))       

    return batch_gt_heatmaps


limbs = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13]]
#可视化关节
def visualize(img, pred_coords):
    """
    :param img: 当前图片
    :param pred_coords: 预测坐标
    :return:
    """
    for joint_idx in range(len(pred_coords)):
        # if gt_coords[joint_idx][0]<0 and gt_coords[joint_idx][0]<0:
        #     continue   
        cv2.circle(img, (int(pred_coords[joint_idx][1]+0.5), int(pred_coords[joint_idx][0]+0.5)), 5, (0, 0, 255), -1)  

    # 画肢体
    for limb_idx in range(len(limbs)):
        # joint_idx1 = limbs[limb_idx][0]    
        # joint_idx2 = limbs[limb_idx][1]     
        # if (gt_coords[joint_idx1][0]<0 and gt_coords[joint_idx1][0]<0) or (gt_coords[joint_idx2][0]<0 and gt_coords[joint_idx2][0]<0):
        #     continue                     
        x1 = pred_coords[limbs[limb_idx][0], 0]  
        y1 = pred_coords[limbs[limb_idx][0], 1]
        x2 = pred_coords[limbs[limb_idx][1], 0]
        y2 = pred_coords[limbs[limb_idx][1], 1]
        cv2.line(img, (int(y1 + 0.5), int(x1 + 0.5)), (int(y2 + 0.5), int(x2 + 0.5)), (0, 255, 0), 2)  


#以人体中心crop图片 targetSize*targetSize 大小
def crop_to_center(orgImgs, targetSize, scales, centers):
    """
    :param orgImgs:
    :param targetSize:
    :param scales:
    :param centers:
    :return:
    """
    output_imgs = []
    for i in range(len(orgImgs)):
        output_img = np.ones((368, 368, 3)) * 128
        img = cv2.imread("/home/administrator/diskb/PengXiao/code/2D-pose/dataset/MPI/images/" + orgImgs[i].decode( 'utf-8'))  

        s = 200 * scales[i] / targetSize 
        img_resized = cv2.resize(img, (0, 0), fx=1 / s, fy=1 / s, interpolation=cv2.INTER_LANCZOS4)  
        center_x = float(np.float64(centers[i][0])) / s
        center_y = float(np.float64(centers[i][1])) / s
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

        output_img[int(fromy0 + 0.5):int(fromy1 + 0.5), int(fromx0 + 0.5):int(fromx1 + 0.5), :] = \
            img_resized[int(y0 + 0.5):int(y1 + 0.5), int(x0 + 0.5):int(x1 + 0.5), :]
        output_imgs.append(output_img)
    return output_imgs


