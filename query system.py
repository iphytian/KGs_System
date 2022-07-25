#!/usr/bin/env python3
# coding: utf-8
# 根据监测系统输入的image quality,输出对应的节点链
# According to the image quality input by the monitoring system, output the corresponding node chain
from py2neo import Graph
import RF_result as svm
import cnn_result as cnn
import imagecutout as cutout
import os
import tensorflow as tf
import numpy as np

# 读取文件夹中的图片
# Read the pictures in the folder
def get_filelist(dir, Filelist):
    if os.path.isfile(dir):
        Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)
    return Filelist

final_result = []

m = 0
numall = 0
Filelist = []
class_result = []
image_path='./dataset_cnn/20191226'  #读取文件存储路径 Read file storage path
dir = image_path
get_filelist(dir, Filelist)
maxnumber = 1

# 读取fit文件最大编号
# Read the maximum number of the fit file
for path in Filelist:
    number = int(path[31:-4])
    if number >= maxnumber:
        maxnumber = number
    else:
        maxnumber = maxnumber

# 循环读取各个相机中的数据得出结果
# Read the data in each camera to get the result
for num in range(maxnumber+1):
    cam0_path = '.\\dataset_cnn\\20191226\\0\\image' + str(num) + '.fit'
    cam1_path = '.\\dataset_cnn\\20191226\\1\\image' + str(num) + '.fit'
    cam2_path = '.\\dataset_cnn\\20191226\\2\\image' + str(num) + '.fit'
    cam3_path = '.\\dataset_cnn\\20191226\\3\\image' + str(num) + '.fit'
    # num0 相机的结果
    # num0 camera result
    if os.path.exists(cam0_path) is False:
        cam0_result = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    else:
        a = svm.svm_clf(cam0_path)
        one_step_result = a[11]
        if one_step_result == 0:
            cam0_result = [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        elif one_step_result == 2:
            cam0_result = [[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        else:
            num = cutout.cutoutimage(cam0_path)
            if num == 0:
                cam0_result = [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            elif num == 1:
                pre = cnn.CNN_clf('.\\dataset_cnn\\0.png')
                pre = np.insert(pre, 0, 0, axis=1)
                pre = np.insert(pre, 0, 0, axis=1)
                cam0_result = tf.add(pre,pre)
            else:
                pre1 = cnn.CNN_clf('.\\dataset_cnn\\0.png')
                pre2 = cnn.CNN_clf('.\\dataset_cnn\\1.png')
                pre3 = tf.add(pre1, pre2)
                pre3 = np.insert(pre3, 0, 0, axis=1)
                pre3 = np.insert(pre3, 0, 0, axis=1)
                cam0_result = pre3

    # num1 相机的结果
    # num1 camera result
    if os.path.exists(cam1_path) is False:
        cam1_result = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    else:
        a = svm.svm_clf(cam1_path)
        one_step_result = a[11]
        if one_step_result == 0:
            cam1_result = [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        elif one_step_result == 2:
            cam1_result = [[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        else:
            num = cutout.cutoutimage(cam1_path)
            if num == 0:
                cam1_result = [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            elif num == 1:
                pre = cnn.CNN_clf('.\\dataset_cnn\\0.png')
                pre = np.insert(pre, 0, 0, axis=1)
                pre = np.insert(pre, 0, 0, axis=1)
                cam1_result = tf.add(pre,pre)
            else:
                pre1 = cnn.CNN_clf('.\\dataset_cnn\\0.png')
                pre2 = cnn.CNN_clf('.\\dataset_cnn\\1.png')
                pre3 = tf.add(pre1, pre2)
                pre3 = np.insert(pre3, 0, 0, axis=1)
                pre3 = np.insert(pre3, 0, 0, axis=1)
                cam1_result = pre3

    # num2 相机的结果
    # num2 camera result
    if os.path.exists(cam2_path) is False:
        cam2_result = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    else:
        a = svm.svm_clf(cam2_path)
        one_step_result = a[11]
        if one_step_result == 0:
            cam2_result = [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        elif one_step_result == 2:
            cam2_result = [[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        else:
            num = cutout.cutoutimage(cam2_path)
            if num == 0:
                cam2_result = [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            elif num == 1:
                pre = cnn.CNN_clf('.\\dataset_cnn\\0.png')
                pre = np.insert(pre, 0, 0, axis=1)
                pre = np.insert(pre, 0, 0, axis=1)
                cam2_result = tf.add(pre,pre)
            else:
                pre1 = cnn.CNN_clf('.\\dataset_cnn\\0.png')
                pre2 = cnn.CNN_clf('.\\dataset_cnn\\1.png')
                pre3 = tf.add(pre1, pre2)
                pre3 = np.insert(pre3, 0, 0, axis=1)
                pre3 = np.insert(pre3, 0, 0, axis=1)
                cam2_result = pre3

    # num3 相机的结果
    # num3 camera result
    if os.path.exists(cam3_path) is False:
        cam3_result = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    else:
        a = svm.svm_clf(cam3_path)
        one_step_result = a[11]
        if one_step_result == 0:
            cam3_result = [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        elif one_step_result == 2:
            cam3_result = [[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        else:
            num = cutout.cutoutimage(cam3_path)
            if num == 0:
                cam3_result = [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            elif num == 1:
                pre = cnn.CNN_clf('.\\dataset_cnn\\0.png')
                pre = np.insert(pre, 0, 0, axis=1)
                pre = np.insert(pre, 0, 0, axis=1)
                cam3_result = tf.add(pre, pre)
            else:
                pre1 = cnn.CNN_clf('.\\dataset_cnn\\0.png')
                pre2 = cnn.CNN_clf('.\\dataset_cnn\\1.png')
                pre3 = tf.add(pre1, pre2)
                pre3 = np.insert(pre3, 0, 0, axis=1)
                pre3 = np.insert(pre3, 0, 0, axis=1)
                cam3_result = pre3

    # 整合四个相机的结果
    # final estimator（Result of integrating four cameras）
    final_result_pre1 = tf.add(cam0_result, cam1_result)
    final_result_pre2 = tf.add(cam2_result, cam3_result)
    final_result_pre3 = tf.add(final_result_pre1, final_result_pre2)
    print(final_result_pre3)
    for i in range(len(final_result_pre3[0])):
        numall = final_result_pre3[0][i] + numall
    if numall == 0:
        final_result = -1
    else:
        final_result_pre4 = tf.argmax(final_result_pre3, axis=1)
        final_result = int(final_result_pre4)

    graph = Graph('http://localhost:7474',user='neo4j',password='19830722')

    #定义匹配结果
    #define the query result
    def find_image_quality(data0):
        image_quality = ''
        if data0 == 0:
            image_quality = 'no_bright_star image'
        elif data0 == 1:
             print('telescope is changing survey area')
        elif data0 == 2:
            image_quality = 'Stick_like image'
        elif data0 == 3:
            image_quality = 'Donut_like image'
        elif data0 == 4:
            image_quality = 'Two_point_like image'
        elif data0 == 5:
            print('telescope is normal')
        else:
            image_quality = 'Lumpy_like image'
        return image_quality

    #根据匹配结果进行查询
    # query based on import
    a = find_image_quality(final_result)
    b = graph.run("MATCH (n{classes:'enviroment'})"
                  "MATCH p=({name:n.name})-[*]->({name:$name}) RETURN n.name, p,length(p)",name = a)

    class_result.append([cam0_path, final_result])