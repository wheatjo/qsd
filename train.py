import os
import argparse
import datetime
import tensorflow as tf
import yolo.config as cfg
from yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc

slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.weights_file ='G:\yolo_tensorflow-master\yolo_tensorflow-master1\yolo_tensorflow-master\data\weights\YOLO_small.ckpt'  #权重文件
        # self.weights_file =None
        self.max_iter = 5000  #15000
        self.initial_learning_rate = cfg.LEARNING_RATE #0.0001
        self.decay_steps = cfg.DECAY_STEPS  #30000
        self.decay_rate = cfg.DECAY_RATE   #衰退率  0.1
        self.staircase = cfg.STAIRCASE  #梯子
        self.summary_iter = cfg.SUMMARY_ITER  # 加起来  10  什么意思
        self.save_iter = 5000 #  ？ 1000
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)   #创一个文件
        self.save_cfg()# 写配置文件
        self.variables = tf.contrib.framework.get_variables_to_restore()
        print(self.variables)

        self.variable_to_restore = [v for v in self.variables if v.name.split('/')[1] != 'fc_36']    #的到一个全局变量
        self.saver = tf.train.Saver(self.variable_to_restore,max_to_keep=1)  #如果你想每训练一代（epoch)就想保存一次模型，则可以将 max_to_keep设置为None或者0
        self.ckpt_file = os.path.join(self.output_dir, 'yolo')  #拼接路径
        self.summary_op = tf.summary.merge_all()# 显示 训练的信息
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)  #画图

        self.global_step = tf.train.create_global_step()  # 画图
        self.learning_rate = tf.train.exponential_decay(          #学习率减小
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(  #梯度 下降
            learning_rate=self.learning_rate)
        self.train_op = slim.learning.create_train_op(
            self.net.total_loss, self.optimizer, global_step=self.global_step) #训练

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(gpu_options=gpu_options)  #分配空间
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        if self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_file)

            # variables = tf.contrib.framework.get_variables_to_restore()
            # variables_to_resotre = [v for v in varialbes if v.name.split('/')[0] != 'fc_36']
            self.saver.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)

    def train(self):

        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self.max_iter + 1):  #15000+1 进行 15000次

            load_timer.tic()
            images, labels = self.data.get()  #得数据
            # print(labels.shape)
            # print(images.shape,labels.shape)
            load_timer.toc()
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:

                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                    log_str = '''{} Epoch: {}, Step: {}, Learning rate: {},
                     Loss: {:5.3f}\nSpeed: {:.3f}s/iter,
                     Load: {:.3f}s/iter, Remain: {}'''.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.data.epoch,
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter))  #输出的信息
                    print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()

                self.writer.add_summary(summary_str, step)

            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.variable_to_restore = [v for v in self.variables ]
                self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=1)
                self.saver.save(
                    self.sess,'yolo/yolo.ckpt',global_step=self.global_step)#, global_step=self.global_step

    def save_cfg(self):#写配置文件

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def update_config_paths(data_dir, weights_file):
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')

    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    if args.data_dir != cfg.DATA_PATH:
        update_config_paths(args.data_dir, args.weights)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet()
    pascal = pascal_voc('train')
    print(pascal)

    solver = Solver(yolo, pascal)  #  模型何 数据

    print('Start training ...')
    solver.train()
    print('Done training.')


if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
