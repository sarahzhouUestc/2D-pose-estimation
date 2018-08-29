import time,os,json,random,cv2
from config import CONFIG
import numpy as np
import h5py
import math

def crop_to_center(orgImg, targetSize, scale, center):
    output_img = np.ones((368, 368, 3)) * 128
    img = cv2.imread("/home/administrator/diskb/PengXiao/code/2D-pose/dataset/MPI/images/" + orgImg.decode('utf-8')) #orgImg是unicode编码

    s = 200 * scale / targetSize   # 相对于368的scale的大小
    img_resized = cv2.resize(img, (0, 0), fx=1/s, fy=1/s, interpolation=cv2.INTER_LANCZOS4)  # 相对于368的1尺度
    center_x = float(np.float64(center[0]))/s
    center_y = float(np.float64(center[1]))/s
    half = targetSize / 2       #目标尺寸的一半

    #图像中的截取的部分
    x0 = max(0, center_x - half)
    y0 = max(0, center_y - half)
    x1 = min(img_resized.shape[1], center_x + half)
    y1 = min(img_resized.shape[0], center_y + half)
    print(int(x0 + 0.5), int(x1 + 0.5), int(y0 + 0.5), int(y1 + 0.5))

    #目标中需要填充的部分
    fromx0 = max(0, half - center_x)
    fromx1 = (img_resized.shape[1] - center_x + half) if (img_resized.shape[1] - center_x) < half else 2 * half
    fromy0 = max(0, half - center_y)
    fromy1 = (half + img_resized.shape[0] - center_y) if (center_y + half) > img_resized.shape[0] else 2 * half
    print(int(fromx0 + 0.5), int(fromx1 + 0.5), int(fromy0 + 0.5), int(fromy1 + 0.5))

    output_img[int(fromy0 + 0.5):int(fromy1 + 0.5), int(fromx0 + 0.5):int(fromx1 + 0.5), :] = \
        img_resized[int(y0 + 0.5):int(y1 + 0.5), int(x0 + 0.5):int(x1 + 0.5), :]
    return output_img
    # cv2.imwrite("/home/administrator/PengXiao/tensorflowtest/2D-pose/test_imgs/out_" + str(time.time()) + ".jpg", output_img)
    # data.close()

if __name__ == '__main__':
    # rotate = CONFIG.augmentation_config['rotate_limit']
    # deg = random.uniform(rotate[0], rotate[1])
    # print(rotate)
    # print(deg)
    # img = cv2.imread("/home/administrator/diskb/PengXiao/code/2D-pose/dataset/MPI/images/028814949.jpg")
    # center = (img.shape[1] * 0.5, img.shape[0] * 0.5)
    # rot_m = cv2.getRotationMatrix2D((int(center[0] + 0.5), int(center[1] + 0.5)), deg, 1)  # 1是图像缩放因子scale, 求得旋转矩阵
    # ret = cv2.warpAffine(img, rot_m, img.shape[1::-1], flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)  # 旋转图像
    # cv2.imwrite("/home/administrator/diskb/PengXiao/code/2D-pose/dataset/MPI/temp/1.jpg", (ret * 255).astype(np.uint8))
    # cv2.waitKey(1000)
    # oriImg = cv2.imread("./test_imgs/princess-diaries-2-00113951.jpg")
    # scale = 368.0 / (oriImg.shape[0] * 1.0)
    # print(scale)
    # imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    #
    # cv2.imwrite("./test_imgs/princess-diaries-2-00113951_resize.jpg", imageToTest)
    # cv2.waitKey(1000)

    data = h5py.File('./dataset/test.h5', 'r')  # 打开h5文件
    print(type(data))
    print(data['imgname'][0])
    img = cv2.imread(CONFIG.image_dir + data['imgname'][0].decode('utf-8'))
    rot_matrix = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), 40, 1)
    print(rot_matrix)
    rot_matrix
    print(type(rot_matrix))
    print(np.matmul(rot_matrix, [10,20,1]))
    print(type(np.matmul(rot_matrix, [10,20,1])))
    x,y = np.matmul(rot_matrix, [10,20,1])
    print(x)
    print(y)

    a = np.mat([[1,0,-img.shape[1]/2], [0,1,-img.shape[0]/2], [0,0,1]])
    b = np.mat([[math.cos(math.pi*40/180), math.sin(math.pi*40/180),0], [-math.sin(math.pi*40/180), math.cos(math.pi*40/180),0], [0,0,1]])
    c = np.mat([[1,0,img.shape[1]/2], [0,1,img.shape[0]/2], [0,0,1]])
    print(np.matmul(c, np.matmul(b,a)))
    # print(b)

    print(10//3)

    x = np.arange(0, 46, 1, float)
    y = x[:, np.newaxis]

    x0 = 10
    y0 = 20

    r = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / 3.0 / 3.0)

    print(x.shape)
    print(y.shape)
    print(x)
    print(y)
    print(((x - x0) ** 2+(y - y0) ** 2).shape)
    print(((x - x0) ** 2).shape)
    print(y-y0)
    print(r.shape)


    # print(list(data.keys()))
    # ['center', 'imgname', 'index', 'normalize', 'part', 'person', 'scale', 'torsoangle', 'visible']
    # 上面的name属性没有用，只有一个值，value为'mpii'
    #  0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck,
    #  9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
    # c = data['index'][idx]
    # k = data['normalize'][idx]
    # e = data['part'][idx]        
    # h = data['torsoangle'][idx]
    # print(h)(b.decode('utf-8'))
    #
    # print(j)
    # a = f['data'][:]  # 取出主键为data的所有的键值

    # crop_to_center(b, 368, s, center)
    # for i in range(len(data['imgname'])):
    #     img = data['imgname'][i]
    #     joints = data['part'][i]
    #     if img.decode('utf-8')=='033944221.jpg':
    #         print("===============")

