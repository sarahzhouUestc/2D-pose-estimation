import numpy as np
#  0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck,
#  9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist

def compute_pck(pnt1, pnt2, gt, preds, thresh):
    """
    :param mesures_rsts: 每人pck预测结果
    :param pnt1: pck: pnt1是右肩  pckh: pnt1是头顶 都是id
    :param pnt2: pnt2是左胯	pnt2是脖子
    :param gt: groudtruth joints
    :param preds: predictive joints
    :return:
    """
    results = []
    if len(gt)!=len(preds): return
    size = len(gt)
    for i in range(size):   
        pred_curr = preds[i]
        gt_curr = gt[i]
        base = np.linalg.norm(np.array(gt_curr[pnt1]) - np.array(gt_curr[pnt2]))  # pck/pckh计算的基准
        res = []
        for j in range(len(gt_curr)):       
            gt_coord = np.array([gt_curr[j][1], gt_curr[j][0]])         
            if gt_coord[0]<=0.0 and gt_coord[1]<=0.0:
                res.append(-1)          
            elif np.linalg.norm(pred_curr[j] - gt_coord) / base <= thresh:
                res.append(1)  
            else:
                res.append(0)  
        results.append(res)
    return results



def compute_pck_accuracy(measures_rsts):     
    """
    :param self:
    :return: 每个关节对应的accuracy列表，平均accuracy
    """
    measures_rsts = np.array(measures_rsts)
    size = len(measures_rsts[0])      
    correct = [sum(measures_rsts[:, i] == 1) for i in range(size)]      
    available = [sum(measures_rsts[:, i] != -1) for i in range(size)]  
    aver_acc = sum(correct)/sum(available)    
    accs = [correct[i]/available[i] for i in range(size)]     
    return accs, aver_acc

