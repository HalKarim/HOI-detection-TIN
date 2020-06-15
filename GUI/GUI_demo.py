# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\GUI_1.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QLabel,QTextBrowser
from PyQt5.Qt import QFileDialog,QGraphicsScene,QImageIOHandler
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QGuiApplication
from PyQt5.QtGui import *
from PyQt5.QtGui import QTransform
from PyQt5.QtCore import Qt
from GUI.GUI_ChildWin import *

from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import scipy.misc
import imageio
import random
import numpy as np
import torch
import torchvision
import pickle
import json
import tensorflow as tf


from models.test_VCOCO_D_pose_pattern_naked import im_detect_GUI_H
from models.test_VCOCO_D_pose_pattern_naked import im_detect_GUI_O
from models.test_VCOCO_D_pose_pattern_naked import im_detect_GUI

from networks.TIN_VCOCO import ResNet50
from lib.ult.vsrl_eval_output_txt import VCOCOeval
from lib.ult.config_vcoco import cfg
from tools.Vcoco_lis_nis import generate_pkl_GUI
from GUI.GUI_MainWin import *
from GUI.GUI_ChildWin import *





class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)
        self.childwin = ChildWindow()

        #self.setGeometry(0, 0, 1024, 600)
        #self.setWindowTitle('main window')

        self.jpg = []
        self.f_HO = True
        self.imgName = []
        self.im_orig = []
        self.im_shape = []

    def selectImage_H(self):
        self.label.flag_ss = True
        self.f_HO = True
        self.pushButton_3.setStyleSheet("background-color: gray")
        self.pushButton_4.setStyleSheet("")
        self.pushButton_5.setStyleSheet("")


    def selectImage_O(self):
        self.label.flag_ss = True
        self.f_HO = False
        self.pushButton_5.setStyleSheet("background-color: gray")
        self.pushButton_4.setStyleSheet("")
        self.pushButton_3.setStyleSheet("")

    def cancelImage(self):
        self.label.x0 = 0
        self.label.x1 = 0
        self.label.y0 = 0
        self.label.y1 = 0
        self.label.update()
        self.label.flag_ss=False
        self.label.flag_s=True
        self.pushButton_3.setStyleSheet("")
        self.pushButton_4.setStyleSheet("background-color: gray")
        self.pushButton_5.setStyleSheet("")

    def openImage(self):
        try:
            imgName, imgType = QFileDialog.getOpenFileName(self.textBrowser, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
            jpg = QtGui.QPixmap(imgName)
            self.textBrowser_2.setText(imgName)
            self.label.setPixmap(jpg)
            self.label.resize(jpg.width(),jpg.height())
            self.imgName=imgName
            self.jpg=jpg
        except:
            self.textBrowser_2.setText("打开文件失败，可能是文件类型错误")


    def DetectionImage(self):
        image_xy=[self.label.x0,self.label.y0,self.label.x1,self.label.y1 ]
        #print(image_xy)
        id=self.imgName[-10:-4]
        id_i=[]
        fg = 0
        for i in id:
            if i != '0' or fg:
                id_i.append(i)
                fg = 1
        id_i=[str(a) for a in id_i]
        image_id= int(''.join(id_i))
        print(self.imgName)

        img = cv2.imread(self.imgName)
        jpg = img[:, :, [2, 1, 0]]
        #im = jpg[self.label.y0:self.label.y1, self.label.x0:self.label.x1]
        #cv2.namedWindow("Image")
        #cv2.imshow("Image", im)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        im_orig = jpg.astype(np.float32, copy=True)
        im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])
        im_shape = im_orig.shape
        #print(im_shape[0], im_shape[1])
        im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
        self.im_orig=im_orig
        self.im_shape=im_shape
        self.jpg=jpg

        #im_orig = jpg.astype(np.float32, copy=True)
        #im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])
        #im_shape = im_orig.shape
        #print(im_shape[0], im_shape[1])
        #im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

        Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_VCOCO_with_pose.pkl', "rb"),
                                encoding='latin1')
        prior_mask = pickle.load(open(cfg.DATA_DIR + '/' + 'prior_mask.pkl', "rb"), encoding='latin1')
        Action_dic = json.load(open(cfg.DATA_DIR + '/' + 'action_index.json'))
        Action_dic_inv = {y: x for x, y in Action_dic.items()}

        vcocoeval = VCOCOeval(cfg.DATA_DIR + '/' + 'v-coco/data/vcoco/vcoco_test.json',
                              cfg.DATA_DIR + '/' + 'v-coco/data/instances_vcoco_all_2014.json',
                              cfg.DATA_DIR + '/' + 'v-coco/data/splits/vcoco_test.ids')

        p_weight = 'E:/projects/Transferable-Interactiveness-Network/Weights/TIN_VCOCO/HOI_iter_6000.ckpt'
        p_output_file = 'E:/projects/Transferable-Interactiveness-Network/-Results/TIN_VCOCO_0.6_0.4_GUItest_naked.pkl' # 最佳预测试结果
        # init session
        tf.reset_default_graph()
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        net = ResNet50()
        net.create_architecture(False)
        saver = tf.train.Saver()
        saver.restore(sess, p_weight)
        p_detection = []

        if self.f_HO:
            im_detect_GUI_H(sess, net, im_orig, im_shape, image_id, image_xy, Test_RCNN, prior_mask, Action_dic_inv,
                            0.4, 0.6, 3, p_detection)
        else:
            im_detect_GUI_O(sess, net, im_orig, im_shape, image_id, image_xy, Test_RCNN, prior_mask, Action_dic_inv,
                            0.4, 0.6, 3, p_detection)
        # print(detection)
        pickle.dump(p_detection, open(p_output_file, "wb"))
        sess.close()
        # 为了得到单个图像的模型预测，对之后的图像test的NIS进行参考

        # weight = 'E:/projects/Transferable-Interactiveness-Network/Weights/TIN_VCOCO/HOI_iter_6000.ckpt'

        weight = 'E:/projects/Transferable-Interactiveness-Network/Weights/TIN_VCOCO_Mytest/HOI_iter_10.ckpt'
        # init session
        tf.reset_default_graph()
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        net = ResNet50()
        net.create_architecture(False)
        saver = tf.train.Saver()
        saver.restore(sess, weight)
        detection = []
        if self.f_HO:
            im_detect_GUI_H(sess, net, im_orig, im_shape, image_id, image_xy, Test_RCNN, prior_mask, Action_dic_inv,
                            0.4,0.6, 3,detection)
        else:
            im_detect_GUI_O(sess, net, im_orig, im_shape, image_id, image_xy, Test_RCNN, prior_mask, Action_dic_inv,
                            0.4,0.6, 3,detection)
        # print(detection)
        # pickle.dump(detection, open(output_file, "wb"))
        sess.close()

        #test_result = detection
        #input_D = cfg.ROOT_DIR + '/-Results/TIN_VCOCO_0.6_0.4_GUItest_naked.pkl'#模型预测，对NIS进行参考
        #with open(input_D, 'rb') as f:
        #    test_D = pickle.load(f, encoding='latin1')
        output_file = 'E:/projects/Transferable-Interactiveness-Network/-Results/p_nis_detection_TIN_VCOCO_0.6_0.4_GUItest_naked.pkl'
        generate_result = generate_pkl_GUI(p_detection, detection, prior_mask, Action_dic_inv, (6, 6, 7, 0), prior_flag=3)
        with open(output_file, 'wb') as f:
           pickle.dump(generate_result, f)

        cc = plt.get_cmap('hsv', lut=6)
        height, width, nbands = jpg.shape

        figsize = width / float(80), height / float(80)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(jpg, interpolation='nearest')

        HO_dic = {}
        HO_set = set()
        count = 0

        if self.f_HO :#框选human，标出object
            for ele in generate_result:
                if (ele['image_id'] == image_id):
                    action_count = -1

                    for action_key, action_value in ele.items():
                        if (action_key.split('_')[
                                -1] != 'agent') and action_key != 'image_id' and action_key != 'person_box' \
                                and action_key != 'object_class' and action_key != 'object_box' \
                                and action_key != 'binary_score'and action_key != 'O_det' and action_key != 'H_det':
                            # print(action_value[:4])
                            # if ele['object_box'] :
                            if (not np.isnan(action_value[0])) and (action_value[4] > 0.01):
                                # print(action_value[:5])
                                O_box = action_value[:4]
                                H_box = ele['person_box']

                                action_count += 1

                                if tuple(O_box) not in HO_set:
                                    HO_dic[tuple(O_box)] = count
                                    HO_set.add(tuple(O_box))
                                    count += 1
                                if tuple(H_box) not in HO_set:
                                    HO_dic[tuple(H_box)] = count
                                    HO_set.add(tuple(H_box))
                                    count += 1

                                '''ax.add_patch(
                                    plt.Rectangle((H_box[0], H_box[1] ),
                                                  H_box[2] - H_box[0],
                                                  H_box[3] - H_box[1], fill=False,
                                                  edgecolor=cc(HO_dic[tuple(H_box)])[:3], linewidth=5)
                                )
                                ax.add_patch(
                                    plt.Rectangle((image_xy[0], image_xy[1]),
                                                  image_xy[2] - image_xy[0],
                                                  image_xy[3] - image_xy[1], fill=False,
                                                  edgecolor=cc(HO_dic[tuple(H_box)])[:3], linewidth=2)
                                )'''
                                text = action_key.split('_')[0] + ', ' + "%.2f" % action_value[4]

                                ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 35,
                                        text,
                                        bbox=dict(facecolor=cc(HO_dic[tuple(O_box)])[:3], alpha=0.5),
                                        fontsize=16, color='white')

                                ax.add_patch(
                                    plt.Rectangle((O_box[0], O_box[1]),
                                                  O_box[2] - O_box[0],
                                                  O_box[3] - O_box[1], fill=False,
                                                  edgecolor=cc(HO_dic[tuple(O_box)])[:3], linewidth=2)
                                )
                                ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
        else :#框选object，标出human
            for ele in generate_result:
                if (ele['image_id'] == image_id):
                    action_count = -1

                    for action_key, action_value in ele.items():
                        if (action_key.split('_')[
                                -1] != 'agent') and action_key != 'image_id' and action_key != 'person_box' \
                                and action_key != 'object_class' and action_key != 'object_box' \
                                and action_key != 'binary_score'and action_key != 'O_det' and action_key != 'H_det':
                            # print(action_value[:4])
                            # if ele['object_box'] :
                            if (not np.isnan(action_value[0])) and (action_value[4] > 0.01):
                                # print(action_value[:5])
                                O_box = action_value[:4]
                                H_box = ele['person_box']

                                action_count += 1

                                if tuple(O_box) not in HO_set:
                                    HO_dic[tuple(O_box)] = count
                                    HO_set.add(tuple(O_box))
                                    count += 1
                                if tuple(H_box) not in HO_set:
                                    HO_dic[tuple(H_box)] = count
                                    HO_set.add(tuple(H_box))
                                    count += 1

                                ax.add_patch(
                                    plt.Rectangle((H_box[0], H_box[1] ),
                                                  H_box[2] - H_box[0],
                                                  H_box[3] - H_box[1], fill=False,
                                                  edgecolor=cc(HO_dic[tuple(H_box)])[:3], linewidth=3)
                                )
                                '''ax.add_patch(
                                    plt.Rectangle((image_xy[0], image_xy[1]),
                                                  image_xy[2] - image_xy[0],
                                                  image_xy[3] - image_xy[1], fill=False,
                                                  edgecolor=cc(HO_dic[tuple(H_box)])[:3], linewidth=2)
                                )'''
                                text = action_key.split('_')[0] + ', ' + "%.2f" % action_value[4]

                                ax.text(H_box[0] + 10, H_box[1] + 25 + action_count * 35,
                                        text,
                                        bbox=dict(facecolor=cc(HO_dic[tuple(O_box)])[:3], alpha=0.5),
                                        fontsize=16, color='white')

                                '''ax.add_patch(
                                    plt.Rectangle((O_box[0], O_box[1]),
                                                  O_box[2] - O_box[0],
                                                  O_box[3] - O_box[1], fill=False,
                                                  edgecolor=cc(HO_dic[tuple(O_box)])[:3], linewidth=2)
                                )'''
                                ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
        plt.show()
        plt.pause(40)
        plt.close()

    def RetrieveImage(self):
        id = self.imgName[-10:-4]
        id_i = []
        fg = 0
        for i in id:
            if i != '0' or fg:
                id_i.append(i)
                fg = 1
        id_i = [str(a) for a in id_i]
        image_id = int(''.join(id_i))
        print(self.imgName)

        img = cv2.imread(self.imgName)
        jpg = img[:, :, [2, 1, 0]]
        im_orig = jpg.astype(np.float32, copy=True)
        im_orig -= np.array([[[102.9801, 115.9465, 122.7717]]])
        im_shape = im_orig.shape
        im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
        self.im_orig = im_orig
        self.im_shape = im_shape
        self.jpg = jpg

        Test_RCNN = pickle.load(open(cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_VCOCO_with_pose.pkl', "rb"),
                                encoding='latin1')
        prior_mask = pickle.load(open(cfg.DATA_DIR + '/' + 'prior_mask.pkl', "rb"), encoding='latin1')
        Action_dic = json.load(open(cfg.DATA_DIR + '/' + 'action_index.json'))
        Action_dic_inv = {y: x for x, y in Action_dic.items()}

        vcocoeval = VCOCOeval(cfg.DATA_DIR + '/' + 'v-coco/data/vcoco/vcoco_test.json',
                              cfg.DATA_DIR + '/' + 'v-coco/data/instances_vcoco_all_2014.json',
                              cfg.DATA_DIR + '/' + 'v-coco/data/splits/vcoco_test.ids')
        p_weight = 'E:/projects/Transferable-Interactiveness-Network/Weights/TIN_VCOCO/HOI_iter_6000.ckpt'
        p_output_file = 'E:/projects/Transferable-Interactiveness-Network/-Results/p_detection_TIN_VCOCO_0.6_0.4_GUItest_naked.pkl'  # 最佳预测试结果
        # init session
        tf.reset_default_graph()
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        net = ResNet50()
        net.create_architecture(False)
        saver = tf.train.Saver()
        saver.restore(sess, p_weight)
        p_detection = []

        im_detect_GUI(sess, net, im_orig, im_shape, image_id, Test_RCNN, prior_mask, Action_dic_inv,
                            0.4, 0.6, 3, p_detection)
        # print(detection)
        pickle.dump(p_detection, open(p_output_file, "wb"))
        sess.close()
        # 为了得到单个图像的模型预测，对之后的图像test的NIS进行参考

        #weight = 'E:/projects/Transferable-Interactiveness-Network/Weights/TIN_VCOCO/HOI_iter_6000.ckpt'

        weight = 'E:/projects/Transferable-Interactiveness-Network/Weights/TIN_VCOCO_Mytest/HOI_iter_10.ckpt'
        # init session
        tf.reset_default_graph()
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        net = ResNet50()
        net.create_architecture(False)
        saver = tf.train.Saver()
        saver.restore(sess, weight)
        detection = []

        im_detect_GUI(sess, net, im_orig, im_shape, image_id, Test_RCNN, prior_mask, Action_dic_inv,
                            0.4, 0.6, 3, detection)
        # print(detection)
        #pickle.dump(detection, open(output_file, "wb"))
        sess.close()

        #test_result = detection
        #input_D = cfg.ROOT_DIR + '/-Results/TIN_VCOCO_0.6_0.4_GUItest_naked.pkl'  # 模型预测，对NIS进行参考
        #with open(input_D, 'rb') as f:
        #    test_D = pickle.load(f, encoding='latin1')
        output_file = 'E:/projects/Transferable-Interactiveness-Network/-Results/p_nis_detection_TIN_VCOCO_0.6_0.4_GUItest_naked.pkl'
        generate_result = generate_pkl_GUI(p_detection, detection, prior_mask, Action_dic_inv, (6, 6, 7, 0), prior_flag=3)
        with open(output_file, 'wb') as f:
            pickle.dump(generate_result, f)
        max = 50
        min = 0
        action_name = []
        action_score = []
        action_HO_weight = []
        for ele in generate_result:
            if (ele['image_id'] == image_id):
                for action_key, action_value in ele.items():
                    if (action_key.split('_')[-1] != 'agent') and action_key != 'image_id' and action_key != 'person_box' \
                            and action_key != 'object_class' and action_key != 'object_box' \
                            and action_key != 'binary_score'and action_key != 'O_det' and action_key != 'H_det':
                        if (not np.isnan(action_value[0])) and (action_value[4] > 0.01):
                            O_box = action_value[:4]
                            H_box = ele['person_box']
                            HO_weight = ((O_box[2] - O_box[0]) * (O_box[3] - O_box[1])) / (
                                    (H_box[2] - H_box[0]) * (H_box[3] - H_box[1]))
                            action_name.append(action_key.split('_')[0])
                            action_score.append(action_value[4])
                            action_HO_weight.append(HO_weight)


        Image_action_MaxScore=np.max(action_score)
        normalization = (Image_action_MaxScore - min) / (max - min)
        Image_action_MaxScore_name=action_name[action_score.index(Image_action_MaxScore)]
        P_HO_weight=action_HO_weight[action_score.index(Image_action_MaxScore)]
        p_similar_score = normalization * 0.5 + 0.5
        #print(action_score)
        #print(Image_action_MaxScore)
        #print(Image_action_MaxScore_name)




        Detection = pickle.load(open(cfg.ROOT_DIR + "/Results/CVPR_best_VCOCO_nis_best_lis.pkl", "rb"))
        image_ids=np.loadtxt(open(cfg.DATA_DIR + '/' + 'v-coco/data/splits/vcoco_test.ids', 'r'))

        D_action_image_id = []
        D_action_image_score = []
        D_action_image_verb_score = []
        for ele in Detection:
            D_action_name = []
            D_action_score = []
            D_action_HO_weight = []
            if ele['image_id'] in image_ids:
                # print(ele['image_id'])
                for action_key, action_value in ele.items():
                    if (action_key.split('_')[-1] != 'agent') and action_key != 'image_id' \
                            and action_key != 'person_box' and action_key != 'object_class' \
                            and action_key != 'object_box' and action_key != 'binary_score'\
                            and action_key != 'O_det' and action_key != 'H_det':
                        if (not np.isnan(action_value[0])) and (action_value[4] > 0.01):
                            O_box = action_value[:4]
                            H_box = ele['person_box']
                            HO_weight = ((O_box[2] - O_box[0]) * (O_box[3] - O_box[1])) / (
                                        (H_box[2] - H_box[0]) * (H_box[3] - H_box[1]))
                            D_action_name.append(action_key.split('_')[0])
                            D_action_score.append(action_value[4])
                            D_action_HO_weight.append(HO_weight)
            if len(D_action_score) == 0:
                continue
            D_Image_action_MaxScore = np.max(D_action_score)
            #max_HO_weight = np.max(D_action_HO_weight)
            D_Image_action_MaxScore_name = D_action_name[D_action_score.index(D_Image_action_MaxScore)]
            D_Image_action_MaxScore_HO_weight = D_action_HO_weight[D_action_score.index(D_Image_action_MaxScore)]
            if D_Image_action_MaxScore_name == Image_action_MaxScore_name:
                D_action_image_id.append(ele['image_id'])
                normalization = (D_Image_action_MaxScore - min) / (max - min)
                similar_score = normalization * 0.2 + ((abs(P_HO_weight - D_Image_action_MaxScore_HO_weight) - 252) ** 2) / 63500 * 0.8 #归一化得分
                D_action_image_score.append(similar_score)
                D_action_image_verb_score.append(D_Image_action_MaxScore)
        #All_Image_action_MaxScore = np.max(D_action_image_score)
        #All_Image_action_MaxScore_id = D_action_image_id[D_action_image_score.index(All_Image_action_MaxScore)]
        new_list_arr = np.array(D_action_image_score)
        list = np.argsort(-new_list_arr)
        All_Image_action_MaxScore_id = []
        All_Image_action_MaxScore_sim = []
        All_Image_action_MaxScore_ver = []
        for i in list:
            im_file =str('E:/DATASETS/VCOCO/v-coco/coco/images/val2014/COCO_val2014_' + (str(D_action_image_id[i]).zfill(12) + '.jpg' ))
            #self.textBrowser.append("<html><p><a href="+im_file+">Image:{0}<br>S_score:{1}<br>V_scroe:{2}</a></p></html>".format(str(D_action_image_id[i]), str('%.4f' % D_action_image_score[i]),str('%.4f' % D_action_image_verb_score[i])))
            self.textBrowser.append("Image:{0}\r\nS_score:{1}\r\nV_scroe:{2}\r\n".format(
                    str(D_action_image_id[i]), str('%.4f' % D_action_image_score[i]),
                    str('%.4f' % D_action_image_verb_score[i])))
            self.cursor = self.textBrowser.textCursor()
            self.textBrowser.moveCursor(self.cursor.End)
            All_Image_action_MaxScore_id.append(D_action_image_id[i])
            All_Image_action_MaxScore_sim.append(D_action_image_score[i])
            All_Image_action_MaxScore_ver.append(D_action_image_verb_score[i])
        self.childwin.show()
        self.childwin.image_show(list,All_Image_action_MaxScore_id,All_Image_action_MaxScore_sim,All_Image_action_MaxScore_ver)



class ChildWindow(QDialog, Ui_dialog):
    def __init__(self):
        super(ChildWindow, self).__init__()
        self.setupUi(self)

        self.setWindowTitle('检索相似图片')

        #self.pushButton.clicked.connect(self.btnClick)  # 按钮事件绑定

    def image_show(self, list,id,sim,ver):
        self.listWidget.setIconSize(QSize(300, 300))
        self.listWidget.setSpacing(10)
        # print(image_ids)
        for i in range(len(list)):
            im_file = 'E:/DATASETS/VCOCO/v-coco/coco/images/val2014/COCO_val2014_' + (str(id[i]).zfill(12) + '.jpg')
            words = '图片ID：' + str(id[i]) + '\r\n相似度得分：' + str(sim[i]) + '\r\n动作得分：' + str(ver[i])
            self.listWidget.addItem(QListWidgetItem(QIcon(im_file), words))
            self.listWidget.show()



if __name__=='__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)                    # Create App
    main = MyMainWindow()
    #ui = Ui_MainWindow()
    #ui.setupUi(MainWinow)
    btn_1 = main.pushButton
    btn_2 = main.pushButton_2
    btn_3 = main.pushButton_3
    btn_4 = main.pushButton_4
    btn_5 = main.pushButton_5
    btn_6 = main.pushButton_6

    btn_1.clicked.connect(main.openImage)
    btn_3.clicked.connect(main.selectImage_H)
    btn_5.clicked.connect(main.selectImage_O)
    btn_4.clicked.connect(main.cancelImage)
    btn_2.clicked.connect(main.DetectionImage)
    btn_6.clicked.connect(main.RetrieveImage)

    child= ChildWindow()
    #child_ui=Ui_dialog()
    #child_ui.setupUi(child)
    #btn_6=ui.pushButton_6
    #btn_6.clicked.connect(child.show)
    main.show()                                  # Show the Window

    sys.exit(app.exec_())

