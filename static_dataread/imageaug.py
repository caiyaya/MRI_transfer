import cv2 as cv
import numpy as np
import os
import xml.etree.ElementTree as ET
import copy
import random




def npy_flip(MRI_npy, mode):
    """
    对 npy二维矩阵进行翻转,旋转
    :param MRI_npy: 待处理的npy文件
    :param mode: 操作方式
    0：左右翻转、1：上下翻转、2：旋转90度、3：旋转180度、4：旋转270度
    :return: 处理后的npy文件
    """
    # 逐列倒序 a[::-1]相当于 a[-1:-len(a)-1-1]:，也就是从最后一个元素到第一个元素复制一遍，即倒序
    e_1 = np.identity(MRI_npy.shape[0], dtype=np.int32)[:, ::-1]
    if mode % 5 == 0:
        img = np.fliplr(MRI_npy)  # 左右翻转
    elif mode % 5 == 1:
        img = np.flipud(MRI_npy)  # 上下翻转
    elif mode % 5 == 2:
        img = MRI_npy.T.dot(e_1)  # .T為轉置函數  順時針旋轉90度
    elif mode % 5 == 3:
        img = e_1.dot(MRI_npy).dot(e_1) # .T為轉置函數  順時針旋轉90度
    elif mode % 5 == 4:
        img = e_1.dot(MRI_npy.T)  # .T為轉置函數  順時針旋轉90度

    return img


def npy_crop(MRI_npy, crop_max=0.4):
    """
    对npy文件进行裁切
    :param MRI_npy: npy文件
    :param crop_max: 至少保留crop_max比例的原图
    :return: 处理后的npy文件
    """
    x_shape = MRI_npy.shape[0]
    y_shape = MRI_npy.shape[1]
    # 判断对一边裁剪或者两边裁剪：0左上,1右下
    x_crop_loc = random.randint(0, 2)  # 生成 [0，2]之间的随机整数
    y_crop_loc = random.randint(0, 2)

    if x_crop_loc == 0:  # 保留 x方向的左半部分
        rand = random.random()  # 生成一个 [0,1)的随机符点数
        x_min = 0
        x_max = x_shape * (1 - crop_max * rand)
    elif x_crop_loc == 1:  # 保留 x方向的右半部分
        rand = random.random()
        x_min = x_shape * (crop_max * rand)
        x_max = x_shape
    else:  # 两边裁剪
        rand1 = random.random()
        rand2 = random.random()
        x_min = x_shape * (crop_max * rand1 / 2)
        x_max = x_shape * (1 - crop_max * rand2 / 2)

    if y_crop_loc == 0:
        rand = random.random()
        y_min = 0
        y_max = y_shape * (1 - crop_max * rand)
    elif x_crop_loc == 1:
        rand = random.random()
        y_min = y_shape * (crop_max * rand)
        y_max = y_shape
    else:
        rand1 = random.random()
        rand2 = random.random()
        y_min = y_shape * (crop_max * rand1 / 2)
        y_max = y_shape * (1 - crop_max * rand2 / 2)

    x_max = int(x_max)
    y_max = int(y_max)
    x_min = int(x_min)
    y_min = int(y_min)

    new_img = np.zeros([x_max - x_min + 2, y_max - y_min + 2])
    # new_img_shape = [y_max - y_min, x_max - x_min]
    new_x = 0
    new_y = 0
    for tmp_i in range(x_min, x_max):
        for tmp_j in range(y_min, y_max):
            new_img[new_x][new_y] = MRI_npy[tmp_i][tmp_j]
            new_y += 1
        new_x += 1
        new_y = 0

    return new_img








# tem_result = tem_result[..., np.newaxis]
# result = np.concatenate((result, tem_result), axis=2)


