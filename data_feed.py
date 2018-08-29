from multiprocessing import Queue, Process
import numpy as np
import data_process
from config import CONFIG


class AssembleData(object):

    class _Sample(object):  
        def __init__(self, img, joints):
            self.img = img
            self.joints = joints

    class _Worker(Process):
        def __init__(self, q, imgname, center, part, scale, coord): 
            Process.__init__(self)
            self.q = q
            self.imgname = imgname
            self.center = center
            self.part = part
            self.scale = scale
            self.total_size = len(center)  
            self.coord = coord

        def run(self):
            while not self.coord.should_stop():             
                idx = np.random.randint(0, self.total_size, 1)  
                output_img, center, output_joints = data_process.resize(self.imgname[idx], self.center[idx], self.part[idx])
                ctc_img, ctc_joints = data_process.crop_to_center(output_img, CONFIG.input_size, self.scale[idx], center, output_joints, True)  
                rotate_img, rotate_joints = data_process.rotate(ctc_img, ctc_joints)
                flip_img, flip_joints = data_process.flip(rotate_img, rotate_joints)
                self.q.put(AssembleData._Sample(flip_img, flip_joints))  

    def __init__(self, q_capacity, data, coord, num_procs, batch_size):
        self.q = Queue(maxsize=q_capacity)
        self.imgname = np.array(data['imgname'])        
        self.center = np.array(data['center'])      
        self.part = np.array(data['part'])          
        self.scale = np.array(data['scale'])        
        self.coord = coord
        self.num_procs = num_procs             
        self.batch_size = batch_size
        self.procs = [AssembleData._Worker(self.q, self.imgname, self.center, self.part, self.scale, self.coord) for _ in range(self.num_procs)]
        self.start_procs()      

    def start_procs(self):
        for p in self.procs:
            p.start()

    def terminate_procs(self):
        for p in self.procs:
            p.join()                

    def get_data(self):
        gt_imgs = []
        gt_joints = []
        while True:
            s = self.q.get()
            gt_imgs.append(s.img)
            gt_joints.append(s.joints)

            if(len(gt_imgs)==self.batch_size):
                yield gt_imgs, gt_joints
                gt_imgs = []
                gt_joints = []

