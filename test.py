import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer

#yaoxu
class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = ['box']  #类    列表
        self.num_class = 1   #多少个类  20
        self.image_size = cfg.IMAGE_SIZE  #448
        self.cell_size = cfg.CELL_SIZE   #7
        self.boxes_per_cell = cfg.BOXES_PER_CELL  #2
        self.threshold = cfg.THRESHOLD  #0.2 置信度的最低值
        self.iou_threshold = cfg.IOU_THRESHOLD  #0.5
        self.boundary1 = self.cell_size * self.cell_size * self.num_class # 7 7 20
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell  #7*7*20    +7*7*2

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)

    def detect(self, img):
        img_h, img_w, _ = img.shape #图片的尺寸
        inputs = cv2.resize(img, (self.image_size, self.image_size)) #调整到预测需要的尺寸
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)   #作用不详
        inputs = (inputs / 255.0) * 2.0 - 1.0  #-1  1
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs}) #进行预测
        results = []  #初始化
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class)) #7 7 2 20
        class_probs = np.reshape(
            output[0:self.boundary1],# 7*7*20
            (self.cell_size, self.cell_size, self.num_class)) #7*7*20
        scales = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell))#7*7* 2
        boxes = np.reshape(
            output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))#7*7*2*4
        offset = np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell)#X轴上的 x的偏置
        offset = np.transpose(
            np.reshape(
                offset,
                [self.boxes_per_cell, self.cell_size, self.cell_size]),#Y轴上的偏置
            (1, 2, 0))

        boxes[:, :, :, 0] += offset  #x轴上恢复偏置
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))#y轴上 恢复偏置
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size #作用不详
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])#W H进行  平方  因为 LOSS是开方的    ????????

        boxes *= self.image_size   #448

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i]) #class的置信度 和本身格子的置信度相乘 7 7 2 20

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')  #进行筛选 小于0.2的为FALSE   7 7 2 20
        filter_mat_boxes = np.nonzero(filter_mat_probs)  #作用不详  返回不为零的下标
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]#筛选  满足条件的全部保存  M * 4
        probs_filtered = probs[filter_mat_probs]  #筛选 M * 1
        classes_num_filtered = np.argmax(
            filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]  #找出最大的类和置信度乘积的概率 的索引

        argsort = np.array(np.argsort(probs_filtered))[::-1]#返回从大到小的数组 2维的
        boxes_filtered = boxes_filtered[argsort] #因为都是2维的  进行第一维度的排列 从大到小
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:  #比较iou
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]  #留下 iou大的
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [self.classes[classes_num_filtered[i]],  #类别的概率
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(
                detect_timer.average_time))

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):
        detect_timer = Timer()
        image = cv2.imread(imname)

        detect_timer.tic()
        result = self.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))

        self.draw_result(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="yolo-5000", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo = YOLONet(False)
    weight_file = 'G:\yolo_tensorflow-master\yolo_tensorflow-master1\yolo_tensorflow-master\data\weights\mmm\yolo.ckpt-5000'  #权重的地址
    detector = Detector(yolo, weight_file)

    # detect from camera
    # cap = cv2.VideoCapture(0)
    # detector.camera_detector(cap)

    # detect from image file
    imname = 'G:\yolo_tensorflow-master\yolo_tensorflow-master1\yolo_tensorflow-master\data\pascal_voc\VOCdevkit\VOC2007\JPEGImages\li.jpg'
    detector.image_detector(imname)


if __name__ == '__main__':
    main()
