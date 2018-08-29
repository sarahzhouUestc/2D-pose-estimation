import tensorflow as tf


class Model(object):
    def __init__(self, stages_num, joints_num):
        """
        :param stages:  stage number
        :param joints:  joints number
        """
        self.stages = stages_num
        self.stage_heatmaps = []        
        self.stage_loss = [0] * stages_num      
        self.total_loss = 0
        self.input_image = None
        self.center_map = None
        self.gt_heatmaps = None
        self.lr = 0
        self.joints = joints_num        
        self.batch_size = 0

    def generate_model(self, input_image, center_map, batch_size):
        self.input_image = input_image
        self.center_map = center_map
        self.batch_size = batch_size
        with tf.variable_scope('pool_center_map', reuse=tf.AUTO_REUSE):
            center_map = tf.layers.average_pooling2d(inputs=self.center_map, pool_size=[9, 9],
                                                    strides=[8, 8], padding='same', name='center_map')
        with tf.variable_scope('sub_stages', reuse=tf.AUTO_REUSE):
            sub_conv1 = tf.layers.conv2d(inputs=input_image,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*368*368*64
                                         name='sub_conv1')
            sub_conv2 = tf.layers.conv2d(inputs=sub_conv1,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*368*368*64
                                         name='sub_conv2')
            sub_pool1 = tf.layers.max_pooling2d(inputs=sub_conv2,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',
                                                name='sub_pool1')       #32*184*184*64
            sub_conv3 = tf.layers.conv2d(inputs=sub_pool1,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*184*184*128
                                         name='sub_conv3')
            sub_conv4 = tf.layers.conv2d(inputs=sub_conv3,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*184*184*128
                                         name='sub_conv4')
            sub_pool2 = tf.layers.max_pooling2d(inputs=sub_conv4,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',
                                                name='sub_pool2')           #32*92*92*128
            sub_conv5 = tf.layers.conv2d(inputs=sub_pool2,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*92*92*256
                                         name='sub_conv5')
            sub_conv6 = tf.layers.conv2d(inputs=sub_conv5,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*92*92*256
                                         name='sub_conv6')
            sub_conv7 = tf.layers.conv2d(inputs=sub_conv6,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*92*92*256
                                         name='sub_conv7')
            sub_conv8 = tf.layers.conv2d(inputs=sub_conv7,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*92*92*256
                                         name='sub_conv8')
            sub_pool3 = tf.layers.max_pooling2d(inputs=sub_conv8,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',
                                                name='sub_pool3')       #32*46*46*256
            sub_conv9 = tf.layers.conv2d(inputs=sub_pool3,
                                         filters=512,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*46*46*512
                                         name='sub_conv9')
            sub_conv10 = tf.layers.conv2d(inputs=sub_conv9,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),    #32*46*46*512
                                          name='sub_conv10')
            sub_conv11 = tf.layers.conv2d(inputs=sub_conv10,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),    #32*46*46*256
                                          name='sub_conv11')
            sub_conv12 = tf.layers.conv2d(inputs=sub_conv11,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),    #32*46*46*256
                                          name='sub_conv12')
            sub_conv13 = tf.layers.conv2d(inputs=sub_conv12,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),    #32*46*46*256
                                          name='sub_conv13')
            sub_conv14 = tf.layers.conv2d(inputs=sub_conv13,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),    #32*46*46*256
                                          name='sub_conv14')

            self.sub_stage_feature_map = tf.layers.conv2d(inputs=sub_conv14,
                                                          filters=128,
                                                          kernel_size=[3, 3],
                                                          strides=[1, 1],
                                                          padding='same',
                                                          activation=tf.nn.relu,
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                          name='sub_stage_feature_map')             #32*46*46*128   以上是提取特征的部分，feature maps

        with tf.variable_scope('stage_1', reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(inputs=self.sub_stage_feature_map,
                                     filters=512,
                                     kernel_size=[1, 1],
                                     strides=[1, 1],
                                     padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*46*46*512
                                     name='conv1')
            self.stage_heatmaps.append(tf.layers.conv2d(inputs=conv1,
                                                        filters=self.joints+1,  
                                                        kernel_size=[1, 1],
                                                        strides=[1, 1],
                                                        padding='same',
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),  #32*46*46*17
                                                        name='stage_heatmap'))
        for stage in range(2, self.stages + 1):
            self._repeated_op(stage, center_map)


    def _repeated_op(self, stage, center_map):          
        with tf.variable_scope('stage_' + str(stage), reuse=tf.AUTO_REUSE):         
            self.current_featuremap = tf.concat([self.stage_heatmaps[stage - 2],            
                                                 self.sub_stage_feature_map,                
                                                 center_map], axis=3)                       
            mid_conv1 = tf.layers.conv2d(inputs=self.current_featuremap,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*46*46*128
                                         name='mid_conv1')
            mid_conv2 = tf.layers.conv2d(inputs=mid_conv1,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*46*46*128
                                         name='mid_conv2')
            mid_conv3 = tf.layers.conv2d(inputs=mid_conv2,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*46*46*128
                                         name='mid_conv3')
            mid_conv4 = tf.layers.conv2d(inputs=mid_conv3,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*46*46*128
                                         name='mid_conv4')
            mid_conv5 = tf.layers.conv2d(inputs=mid_conv4,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*46*46*128
                                         name='mid_conv5')
            mid_conv6 = tf.layers.conv2d(inputs=mid_conv5,
                                         filters=128,
                                         kernel_size=[1, 1],                            
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),     #32*46*46*128
                                         name='mid_conv6')
            self.current_heatmap = tf.layers.conv2d(inputs=mid_conv6,
                                                    filters=self.joints+1,
                                                    kernel_size=[1, 1],         
                                                    strides=[1, 1],
                                                    padding='same',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),  #32*46*46*17
                                                    name='mid_conv7')
            self.stage_heatmaps.append(self.current_heatmap)


    def generate_loss(self, gt_heatmaps, lr, lr_decay_rate, lr_decay_steps):
        self.gt_heatmaps = gt_heatmaps
        self.total_loss = 0
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate          
        self.lr_decay_steps = lr_decay_steps        

        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss', reuse=tf.AUTO_REUSE):         
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmaps[stage] - self.gt_heatmaps,  name='l2_loss') / self.batch_size    

        with tf.variable_scope('total_loss', reuse=tf.AUTO_REUSE):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]           

        with tf.variable_scope('training', reuse=tf.AUTO_REUSE):
            self.global_step = tf.contrib.framework.get_or_create_global_step()         

            self.lr = tf.train.exponential_decay(self.lr, global_step=self.global_step,         
                                                 decay_rate=self.lr_decay_rate,
                                                 decay_steps=self.lr_decay_steps)

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss, global_step=self.global_step,
                                                            learning_rate=self.lr, optimizer='Adam')        
