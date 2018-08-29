import tensorflow as tf
import numpy as np
import cv2
import os
from models import sppe
from utils import utils
from config import CONFIG
from data_feed import AssembleData
import h5py
import evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_data_file', default_value='./dataset/train.h5', docstring="train data file path")
tf.app.flags.DEFINE_integer('train_input_size', default_value=368, docstring="input image size")
tf.app.flags.DEFINE_integer('heatmap_size', default_value=46, docstring='heatmap size')
tf.app.flags.DEFINE_integer('centermap_var', default_value=50, docstring='center map gaussian variance')
tf.app.flags.DEFINE_integer('joints_num', default_value=16, docstring='number of joints')
tf.app.flags.DEFINE_integer('stages_num', default_value=6, docstring="model stages number")
tf.app.flags.DEFINE_integer('train_batch_size', default_value=32, docstring="batch size")
tf.app.flags.DEFINE_float('init_lr', default_value=0.0003, docstring="initialize learning rate")
tf.app.flags.DEFINE_float('lr_decay_rate', default_value=0.95, docstring="learning rate decay rate")
tf.app.flags.DEFINE_integer('lr_decay_steps', default_value=100, docstring="learning rate decay steps")
tf.app.flags.DEFINE_integer('ckpt_iters', default_value=10, docstring="save models every ## iterations")
tf.app.flags.DEFINE_integer('train_iters', default_value=20000, docstring="total training iterations")
tf.app.flags.DEFINE_integer('middle_rst_iters', default_value=10, docstring="save middle results every ## iterations")



def main(argv):

    """
    Model
    """
    model = sppe.Model(FLAGS.stages_num, FLAGS.joints_num)

    img_input = tf.placeholder(dtype=tf.float32, shape=[FLAGS.train_batch_size, FLAGS.train_input_size, FLAGS.train_input_size, 3], name="input_image")
    gt_heatmap = tf.placeholder(dtype=tf.float32, shape=[FLAGS.train_batch_size, 46,46, FLAGS.joints_num+1], name='gt_heatmap')
    center_map = tf.placeholder(dtype=tf.float32, shape=[FLAGS.train_batch_size, FLAGS.train_input_size, FLAGS.train_input_size, 1], name='center_map')

    model.generate_model(img_input, center_map, FLAGS.train_batch_size)
    model.generate_loss(gt_heatmap, FLAGS.init_lr, FLAGS.lr_decay_rate, FLAGS.lr_decay_steps)

    """ 
    data feed
    """
    coord = tf.train.Coordinator()      
    data = h5py.File(FLAGS.train_data_file, 'r')  
    assemble = AssembleData(1000, data, coord, 10, 32)      
    g = assemble.get_data()     

    """ 
    Training
    """
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:      
        saver = tf.train.Saver()            
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for train_itr in range(FLAGS.train_iters):          
            # 读取一个batch的数据
            batch_imgs, batch_gt_joints = next(g)       
            batch_imgs = batch_imgs / 255.0 - 0.5         
            batch_gt_heatmaps = utils.generate_heatmaps_from_joints(FLAGS.train_input_size, FLAGS.heatmap_size, CONFIG.hm_gaussian_variance, batch_gt_joints)     

            # 构造center_map
            maps = []
            for i in range(len(batch_imgs)):       
                map = utils.generate_gaussian_map(FLAGS.train_input_size, FLAGS.train_input_size, FLAGS.train_input_size / 2, FLAGS.train_input_size / 2, FLAGS.centermap_var)  
                map = np.reshape(map, [FLAGS.train_input_size, FLAGS.train_input_size, 1])
                maps.append(map)
            maps = np.array(maps)      


            stage_losses, total_loss, _,  current_lr, stage_heatmaps, global_step = \
                sess.run([model.stage_loss,
                          model.total_loss,
                          model.train_op,
                          model.lr,
                          model.stage_heatmaps,
                          model.global_step
                          ], feed_dict={model.input_image: batch_imgs,
                                        model.gt_heatmaps: batch_gt_heatmaps,
                                        model.center_map: maps})

            # 打印日志
            print_training_status(global_step, current_lr, stage_losses, total_loss)

            # 保存中间结果
            if global_step % FLAGS.middle_rst_iters == 0:      
                img = batch_imgs[0] + 0.5         

                save_stage_heatmaps = []
                for stage in range(FLAGS.stages_num):
                    save_stage_heatmap = stage_heatmaps[stage][0, :, :, 0:FLAGS.joints_num].reshape(
                                        (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.joints_num))     #(46,46,16)
                    save_stage_heatmap = cv2.resize(save_stage_heatmap, (FLAGS.train_input_size, FLAGS.train_input_size))      
                    save_stage_heatmap = np.amax(save_stage_heatmap, axis=2)        
                    save_stage_heatmap = np.reshape(save_stage_heatmap, (FLAGS.train_input_size, FLAGS.train_input_size, 1))
                    save_stage_heatmap = np.repeat(save_stage_heatmap, 3, axis=2)       
                    save_stage_heatmaps.append(save_stage_heatmap)

                save_gt_heatmap = batch_gt_heatmaps[0, :, :, 0:FLAGS.joints_num].reshape(
                                        (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.joints_num))    
                save_gt_heatmap = cv2.resize(save_gt_heatmap, (FLAGS.train_input_size, FLAGS.train_input_size))
                save_gt_heatmap = np.amax(save_gt_heatmap, axis=2)
                save_gt_heatmap = np.reshape(save_gt_heatmap, (FLAGS.train_input_size, FLAGS.train_input_size, 1))
                save_gt_heatmap = np.repeat(save_gt_heatmap, 3, axis=2)

                upper_img = np.concatenate((save_stage_heatmaps[0], save_stage_heatmaps[1], save_stage_heatmaps[2]), axis=1)   
                overlap_img = 0.5 * img + 0.5 * save_gt_heatmap         
                lower_img = np.concatenate((save_stage_heatmaps[FLAGS.stages_num - 1], save_gt_heatmap, overlap_img), axis=1)   
                save_img = np.concatenate((upper_img, lower_img), axis=0)      
                cv2.imwrite(os.path.join(CONFIG.middle_output, str(global_step) + ".jpg"), (save_img * 255).astype(np.uint8))

            
            if (global_step + 1) % FLAGS.ckpt_iters == 0:        
                saver.save(sess=sess, save_path=os.path.join(CONFIG.model_dir, CONFIG.model_name), global_step=(global_step + 1))

        coord.request_stop()
        assemble.terminate_procs()          
    print('Training done.')


def print_training_status(global_step, cur_lr, stage_losses, total_loss):
    status = 'Step: {}/{} ----- Cur_lr: {:1.7f} '.format(global_step, FLAGS.train_iters, cur_lr)
    losses = ' | '.join(['S{} loss: {:7.2f}'.format(stage_num + 1, stage_losses[stage_num]) for stage_num in range(FLAGS.stages_num)])
    losses += ' | Total loss: {}'.format(total_loss)
    print(status)
    print(losses + '\n')


if __name__ == '__main__':
    tf.app.run()