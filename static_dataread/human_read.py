import glob
import os
import sys
import numpy as np
import random
import collections
from imblearn.over_sampling import RandomOverSampler
from static_dataread.unaligned_data_loader import UnalignedDataLoader
from utils import *
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import time
"""
多模态版
"""

def concatenate_along_first_axis(arr_list):
    # 使用 numpy.concatenate 沿第一维拼接多个数组
    concatenated_arr = np.concatenate(arr_list, axis=0)

    return concatenated_arr


# -------------------- 针对每个模态一张的版本 --------------------- #

# 标签是几就是几分类版
def return_dataset(data_path, label_path, modality, slice_size):
    """.
    从data_path和label_path中分别读取numpy数据和 label, modality是每个病人取的模态数
        return:
            data:所有病例的numpy数据（三维数组）
            label0: 纤维化标签（0-4）  (int)
            label1: 脂变标签（0-3）  (int)
            label2: 炎症标签（0-5）  (int)
            order_num: 每例病例的切片张数  (int)
            height: 单张切片的长  (int)
            width: 单张切片的宽  (int)
            modality  # 模态数
    """
    data_name_list = os.listdir(data_path)
    label_name_list = os.listdir(label_path)
    data_list = []  # 存储数据文件所在路径
    label_list = []  # 存储label文件所在路径
    label_str = []
    data = []  # 存储数据文件的内容（数组）
    label0 = np.empty(shape=0)  # 空数组
    label1 = np.empty(shape=0)
    label2 = np.empty(shape=0)
    sample_num = 0

    # ------------------------已改为多模态-------------------------
    # data_path = r'D:\Coding\my_code\EMCA_net\data\human_mm_data'
    for human in data_name_list:
        tmp_path = os.path.join(data_path, human)  # os.sep是文件路径里的反斜杠，tmp_path是list型
        for mode in os.listdir(tmp_path):
            data_list.append(data_path + os.sep + human + os.sep + mode)
        # print("----", tmp_path, " is join in data_list----")
    # -----------------------------------------------------------

    for human in label_name_list:
        for mode in os.listdir(label_path + os.sep + human):
            label_list.append(label_path + os.sep + human + os.sep + mode)

    for i in data_list:
        d = np.load(i, allow_pickle=True)
        # print(d.shape)
        d = d.reshape([slice_size, slice_size])
        data.append(d)
        sample_num += 1  # 总切片张数
    data = np.array(data)
    height = data.shape[1]
    width = data.shape[2]
    data = data.reshape((sample_num // modality, modality * height * width))  # 将数组整理成 病人*单张大切片的shape

    # 读取label.txt内的信息
    for i in label_list:
        with open(i, 'r') as f:
            for line in f:  # 按行读取
                label_str.append(list(line.split(',')))  # label_str：以','为分割读成list插入
    for la in label_str:
        # 每个标签都以0为正常值

        label0 = np.append(label0, int(la[0]))  # 纤维化
        label1 = np.append(label1, int(la[1]) - 5)  # 脂变
        label2 = np.append(label2, int(la[2]) - 9)  # 炎症

    return data, label0, label1, label2, modality, height, width


# 纤维化二分类版
def return_dataset_2(data_path, label_path, modality, slice_size):
    """.
    从data_path和label_path中分别读取numpy数据和 label, modality是每个病人取的模态数
        return:
            data:所有病例的numpy数据（三维数组）
            label0: 纤维化标签（0-4）  (int)
            label1: 脂变标签（0-3）  (int)
            label2: 炎症标签（0-5）  (int)
            order_num: 每例病例的切片张数  (int)
            height: 单张切片的长  (int)
            width: 单张切片的宽  (int)
            modality  # 模态数
    """
    # data_name_list = os.listdir(data_path)
    # label_name_list = os.listdir(label_path)
    data_name_list = data_path
    label_name_list = label_path
    data_list = []  # 存储数据文件所在路径
    label_list = []  # 存储label文件所在路径
    label_str = []
    data = []  # 存储数据文件的内容（数组）
    label0 = np.empty(shape=0)  # 空数组
    label1 = np.empty(shape=0)
    label2 = np.empty(shape=0)
    sample_num = 0

    # ------------------------已改为多模态-------------------------
    # data_path = r'D:\Coding\my_code\EMCA_net\data\human_mm_data'
    for human in data_name_list:
        # tmp_path = os.path.join(data_path, human)  # os.sep是文件路径里的反斜杠，tmp_path是list型
        # if os.path.isdir(tmp_path):
        #     for mode in os.listdir(tmp_path):
        #         data_list.append(data_path + os.sep + human + os.sep + mode)
        if os.path.isdir(human):
            for mode in os.listdir(human):
                data_list.append(os.path.join(human, mode))
        # print("----", tmp_path, " is join in data_list----")
    # -----------------------------------------------------------

    for human in label_name_list:
        # if os.path.isdir(tmp_path):
        #     for mode in os.listdir(label_path + os.sep + human):
        #         label_list.append(label_path + os.sep + human + os.sep + mode)
        if os.path.isdir(human):
            for mode in os.listdir(human):
                label_list.append(os.path.join(human, mode))

    for i in data_list:
        d = np.load(i, allow_pickle=True)
        # print(d.shape)
        d = d.reshape([slice_size, slice_size])
        data.append(d)
        sample_num += 1  # 总切片张数
    data = np.array(data)
    height = data.shape[1]
    width = data.shape[2]
    data = data.reshape((sample_num // modality, modality * height * width))  # 将数组整理成 病人*单张大切片的shape

    # 读取label.txt内的信息
    for i in label_list:
        with open(i, 'r') as f:
            for line in f:  # 按行读取
                label_str.append(list(line.split(',')))  # label_str：以','为分割读成list插入
    for la in label_str:
        # 每个标签都以0为正常值
        if int(la[0]) < 3:
            label0 = np.append(label0, 0)  # 纤维化
        else:
            label0 = np.append(label0, 1)  # 纤维化
        # label0 = np.append(label0, int(la[0]))  # 纤维化
        label1 = np.append(label1, int(la[1]) - 5)  # 脂变
        label2 = np.append(label2, int(la[2]) - 9)  # 炎症

    print("data.shape = ", data.shape)

    return data, label0, label1, label2, modality, height, width


# 标签是几就是几分类版
def return_dataset_train(data_name_list,label_name_list, modality, slice_size):
    """
    从data_path和label_path中分别读取numpy数据和 label, modality是每个病人取的模态数
        return:
            data:所有病例的numpy数据（三维数组）
            label0: 纤维化标签（0-4）  (int)
            label1: 脂变标签（0-3）  (int)
            label2: 炎症标签（0-5）  (int)
            order_num: 每例病例的切片张数  (int)
            height: 单张切片的长  (int)
            width: 单张切片的宽  (int)
            modality  # 模态数
    """
    data_name_list = data_name_list
    label_name_list = label_name_list
    data_list = []  # 存储数据文件所在路径
    label_list = []  # 存储label文件所在路径
    label_str = []
    data = []  # 存储数据文件的内容（数组）
    label0 = np.empty(shape=0)  # 空数组
    label1 = np.empty(shape=0)
    label2 = np.empty(shape=0)
    sample_num = 0

    # ------------------------已改为多模态-------------------------
    # data_path = r'D:\Coding\my_code\EMCA_net\data\human_mm_data'

    for data_path in data_name_list:
        tmp_data_path = os.listdir(data_path)
        tmp_data_path.sort()
        for mode in tmp_data_path:
            data_list.append(data_path + os.sep + mode)
        # print("----", tmp_path, " is join in data_list----")
    # -----------------------------------------------------------
    for label_path in label_name_list:
        for mode in os.listdir(label_path):
            label_list.append(label_path + os.sep + mode)

    for i in data_list:
        d = np.load(i, allow_pickle=True)
        # print(d.shape)
        # d = d.reshape([slice_size, slice_size])
        data.append(d)
        sample_num += 1  # 总切片张数
    # data = np.array(data)
    data = np.stack(data, axis=0)
    height = data.shape[1]
    width = data.shape[2]
    data = data.reshape((sample_num // modality, modality * height * width))  # 将数组整理成 病人*单张大切片的shape

    # 读取label.txt内的信息
    for i in label_list:
        with open(i, 'r') as f:
            for line in f:  # 按行读取
                label_str.append(list(line.split(',')))  # label_str：以','为分割读成list插入
    for la in label_str:
        # 每个标签都以0为正常值
        label0 = np.append(label0, int(la[0]))  # 纤维化
        label1 = np.append(label1, int(la[1]) - 5)  # 脂变
        label2 = np.append(label2, int(la[2]) - 9)  # 炎症


    return data, label0, label1, label2, modality, height, width


# 二分类版
def return_dataset_train_2(data_name_list,label_name_list, modality, slice_size):
    """
    从data_path和label_path中分别读取numpy数据和 label, modality是每个病人取的模态数
        return:
            data:所有病例的numpy数据（三维数组）
            label0: 纤维化标签（0-4）  (int)
            label1: 脂变标签（0-3）  (int)
            label2: 炎症标签（0-5）  (int)
            order_num: 每例病例的切片张数  (int)
            height: 单张切片的长  (int)
            width: 单张切片的宽  (int)
            modality  # 模态数
    """
    data_name_list = data_name_list
    label_name_list = label_name_list
    data_list = []  # 存储数据文件所在路径
    label_list = []  # 存储label文件所在路径
    label_str = []
    label0 = np.empty(shape=0)  # 空数组
    label1 = np.empty(shape=0)
    label2 = np.empty(shape=0)
    data = []
    sample_num = 0

    # ------------------------已改为多模态-------------------------
    # data_path = r'D:\Coding\my_code\EMCA_net\data\human_mm_data'

    for data_path in data_name_list:
        tmp_data_path = os.listdir(data_path)
        tmp_data_path.sort()
        for mode in tmp_data_path:
            # 这里读取的mice_190下的npy文件命名为byte类型 需要转换为str类型
            if isinstance(mode, bytes):
                data_list.append(os.path.join(data_path, mode.decode('utf-8')))
            else:
                data_list.append(os.path.join(data_path, mode))

        # print("----", tmp_path, " is join in data_list----")
    # -----------------------------------------------------------
    for label_path in label_name_list:
        for mode in os.listdir(label_path):
            label_list.append(label_path + os.sep + mode)
    # 每个模态一张
    for i in data_list:
        d = np.load(i, allow_pickle=True)
        # 检查d的维度
        if d.shape != (128, 128):
            d = np.squeeze(d)
            print("error in {}, {}".format(i, d.shape))
        data.append(d)
        sample_num += 1  # 总切片张数
    data = np.array(data)
    height = data.shape[1]
    width = data.shape[2]
    data = data.reshape((sample_num // modality, modality * height * width))  # 将数组整理成 病人*单张大切片的shape

    # 每个模态五张
    # for i in data_list:
    #     d = np.load(i, allow_pickle=True)
    #     for d_slice in range(d.shape[0]):
    #         data.append(d[d_slice])
    #         sample_num += 1  # 总切片张数
    # data_npy = np.stack(data, axis=0)
    #
    # height = data.shape[1]
    # width = data.shape[2]
    # data_npy = data_npy.reshape((sample_num // modality, modality * height * width))  # 将数组整理成 病人*单张大切片的shape



    # 读取label.txt内的信息
    for i in label_list:
        with open(i, 'r') as f:
            for line in f:  # 按行读取
                label_str.append(list(line.split(',')))  # label_str：以','为分割读成list插入
    for la in label_str:
        # 每个标签都以0为正常值
        if int(la[0]) < 3:
            label0 = np.append(label0, 0)  # 纤维化
        else:
            label0 = np.append(label0, 1)  # 纤维化
        # label0 = np.append(label0, int(la[0]))  # 纤维化
        label1 = np.append(label1, int(la[1]) - 5)  # 脂变
        label2 = np.append(label2, int(la[2]) - 9)  # 炎症


    return data, label0, label1, label2, modality, height, width



def max_num_counter(label, class_num):
    """
    得到实际读取到的 label中最大的一个label值
    input: label：所有病例的label数组   class_num：类别数
    return: max_num：最大的label值
    """
    class_num_list = [0] * class_num  # 新建列表
    label_list = label.tolist()  # 将矩阵或数组转换为list
    dict_label = collections.Counter(label_list)  # 将 label_list 转为一个字典
    i = 0
    for s in dict_label:
        class_num_list[i] = dict_label[s]
        i = i + 1
    max_num = max(class_num_list)
    print("class_num_list:", dict(zip(dict_label, class_num_list)))
    return max_num


def oversampling(source_data, source_label, target_data, target_label, class_num):
    """
    如何调包：https://github.com/scikit-learn-contrib/imbalanced-learn

    ROS(Random Over-sampling):因为医疗数据不能随便伪造，医生尚且不承认用GAN，
                              更不必说SMOTE等传统oversampling的方法了，
                              完全有可能破坏影像数据本身自带的结构信息，
                              所以还是用ROS，随机重复地选取某些样本。
    函数文档：https://imbalanced-learn.org/stable/generated/imblearn.over_sampling.RandomOverSampler.html#imblearn.over_sampling.RandomOverSampler
    具体实现：https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/over_sampling/_random_over_sampler.py
    """
    source_max_num = max_num_counter(source_label, class_num)
    target_max_num = max_num_counter(target_label, class_num)
    # 找出源域和目标域标签更全的那个  max_num：最大标签值
    if source_max_num > target_max_num:
        max_num = source_max_num
    else:
        max_num = target_max_num

    ROS_strategy = {x: max_num for x in range(class_num)}  # 字典
    ros = RandomOverSampler(sampling_strategy=ROS_strategy)
    xs_ROS, ys_ROS = ros.fit_resample(source_data, source_label)
    xt_ROS, yt_ROS = ros.fit_resample(target_data, target_label)

    """
    return: 
        xs_ROS:源域重采样后生成的矩阵；ys_ROS：xs_ROS对应的label
        xt_ROS:目标域重采样后生成的矩阵；ys_ROS：xt_ROS对应的label
    """
    return xs_ROS, ys_ROS, xt_ROS, yt_ROS


def read_dataset(source_dir, label_s_dir, target_dir, label_t_dir, slice_size, modality, class_num, mode):
    # 二分类版
    xs, ys1, ys2, ys3, s_order, s_height, s_width = return_dataset_2(source_dir, label_s_dir, modality=modality, slice_size=slice_size)
    xt, yt1, yt2, yt3, t_order, t_height, t_width = return_dataset_2(target_dir, label_t_dir, modality=modality, slice_size=slice_size)

    # 五分类版
    # xs, ys1, ys2, ys3, s_order, s_height, s_width = return_dataset(source_dir, label_s_dir, modality=modality,
    #                                                                  slice_size=slice_size)
    # xt, yt1, yt2, yt3, t_order, t_height, t_width = return_dataset(target_dir, label_t_dir, modality=modality, slice_size=slice_size)

    if mode == 0:  # 对纤维化进行采样
        xs_ROS, ys_ROS, xt_ROS, yt_ROS = xs, ys1, xt, yt1
    if mode == 1:  # 对脂变进行采样
        xs_ROS, ys_ROS, xt_ROS, yt_ROS = xs, ys2, xt, yt2
    if mode == 2:  # 对炎症进行采样
        xs_ROS, ys_ROS, xt_ROS, yt_ROS = xs, ys3, xt, yt3

    xs_sample_num = xs_ROS.shape[0]  # 病例数
    xt_sample_num = xt_ROS.shape[0]

    # 判断源域和目标域的尺寸是否相同（order: 切片张数）
    # if (s_order == t_order) & (s_height == t_height) & (s_width == t_width) & (xs_sample_num == xt_sample_num):
    #     order = s_order
    #     height = s_height
    #     width = s_width
    #     sample_num = xs_sample_num
    # else:
    #     print("The shape is not identical")

    order = s_order
    height = s_height
    width = s_width

    xs_ROS = xs_ROS.reshape((xs_sample_num, order, height, width))  # 重新调整尺寸
    xt_ROS = xt_ROS.reshape((xt_sample_num, order, height, width))

    return xs_ROS, ys_ROS, xt_ROS, yt_ROS


def KflodDataloader(slice_size, mo_channel, class_num, mode, source_dir, label_s_dir, target_dir, label_t_dir, target_v_dir, label_tv_dir):
    """
    :param class_num: 类别数
    :param mode: 0：纤维化；1：脂变；2：炎症
    :return: 源域数据，源域标签（one-hot），目标域数据，目标域标签（one-hot），病例数
    """
    start_time = time.time()

    # 二分类版
    xs, ys1, ys2, ys3, s_order, s_height, s_width = return_dataset_train_2(source_dir, label_s_dir, modality=mo_channel, slice_size=slice_size)
    xt, yt1, yt2, yt3, t_order, t_height, t_width = return_dataset_train_2(target_dir, label_t_dir, modality=mo_channel, slice_size=slice_size)

    xtv, ytv1, ytv2, ytv3, tv_order, tv_height, tv_width = return_dataset_train_2(target_v_dir, label_tv_dir, modality=mo_channel, slice_size=slice_size)

    # 埋点6
    end_time = time.time()  # 获取结束时间
    print(f"return_dataset_train_2数据加载运行时间: {end_time - start_time} 秒")
    start_time = time.time()

    # 同时也要对 xtv 和 ytv 进行采样 采样参考 valid部分
    if mode == 0:  # 对纤维化进行采样
        xtv_ROS, ytv_ROS = xtv, ytv1
    if mode == 1:  # 对脂变进行采样
        xtv_ROS, ytv_ROS = xtv, ytv2
    if mode == 2:  # 对炎症进行采样
        xtv_ROS, ytv_ROS = xtv, ytv3

    xtv_sample_num = xtv_ROS.shape[0]

    if mode == 0:  # 对纤维化进行采样
        xs_ROS, ys_ROS, xt_ROS, yt_ROS = oversampling(xs, ys1, xt, yt1, class_num)
    if mode == 1:  # 对脂变进行采样
        xs_ROS, ys_ROS, xt_ROS, yt_ROS = oversampling(xs, ys2, xt, yt2, class_num)
    if mode == 2:  # 对炎症进行采样
        xs_ROS, ys_ROS, xt_ROS, yt_ROS = oversampling(xs, ys3, xt, yt3, class_num)

    xs_sample_num = xs_ROS.shape[0]
    xt_sample_num = xt_ROS.shape[0]


    order = s_order
    height = s_height
    width = s_width
    xs_ROS = xs_ROS.reshape((xs_sample_num, order, height, width))  # 重新调整尺寸
    xt_ROS = xt_ROS.reshape((xt_sample_num, order, height, width))
    xtv_ROS = xtv_ROS.reshape((xtv_sample_num, order, height, width))
    # print("Reshape finish!")

    # 埋点7
    end_time = time.time()  # 获取结束时间
    print(f"各种reshape数据运行时间: {end_time - start_time} 秒")
    start_time = time.time()

    return xs_ROS, ys_ROS, xt_ROS, yt_ROS, xtv_ROS, ytv_ROS

def read_dataset_train(slice_size, mo_channel, class_num, mode, source_dir, label_s_dir, target_dir, label_t_dir):
    """
    :param class_num: 类别数
    :param mode: 0：纤维化；1：脂变；2：炎症
    :return: 源域数据，源域标签（one-hot），目标域数据，目标域标签（one-hot），病例数
    """
    # print("Source data Loading......")
    # xs.shape = (病例数，切片数，宽，高)

    # 二分类版
    xs, ys1, ys2, ys3, s_order, s_height, s_width = return_dataset_train_2(source_dir, label_s_dir, modality=mo_channel, slice_size=slice_size)
    xt, yt1, yt2, yt3, t_order, t_height, t_width = return_dataset_train_2(target_dir, label_t_dir, modality=mo_channel, slice_size=slice_size)


    # 五分类版
    # xs, ys1, ys2, ys3, s_order, s_height, s_width = return_dataset_train(source_dir, label_s_dir, modality=mo_channel,
    #                                                                        slice_size=slice_size)
    # xt, yt1, yt2, yt3, t_order, t_height, t_width = return_dataset_train(target_dir, label_t_dir, modality=mo_channel,
    #                                                                        slice_size=slice_size)

    if mode == 0:  # 对纤维化进行采样
        xs_ROS, ys_ROS, xt_ROS, yt_ROS = oversampling(xs, ys1, xt, yt1, class_num)
    if mode == 1:  # 对脂变进行采样
        xs_ROS, ys_ROS, xt_ROS, yt_ROS = oversampling(xs, ys2, xt, yt2, class_num)
    if mode == 2:  # 对炎症进行采样
        xs_ROS, ys_ROS, xt_ROS, yt_ROS = oversampling(xs, ys3, xt, yt3, class_num)

    xs_sample_num = xs_ROS.shape[0]
    xt_sample_num = xt_ROS.shape[0]

    # 判断源域和目标域的尺寸是否相同（order: 切片张数）
    # if (s_order == t_order) & (s_height == t_height) & (s_width == t_width) & (xs_sample_num == xt_sample_num):
    #     order = s_order
    #     height = s_height
    #     width = s_width
    #     sample_num = xs_sample_num
    # else:
    #     print("The shape is not identical")

    order = s_order
    height = s_height
    width = s_width
    xs_ROS = xs_ROS.reshape((xs_sample_num, order, height, width))  # 重新调整尺寸
    xt_ROS = xt_ROS.reshape((xt_sample_num, order, height, width))
    # print("Reshape finish!")

    return xs_ROS, ys_ROS, xt_ROS, yt_ROS


def read_dataset_valid(slice_size, modality, class_num, mode,source_dir, label_s_dir, target_dir, label_t_dir):
    """
    :param class_num: 类别数
    :param mode: 0：纤维化；1：脂变；2：炎症
    :param times: 数据扩充倍数
    :return: 源域数据，源域标签（one-hot），目标域数据，目标域标签（one-hot），病例数
    """
    # print("Source data Loading......")
    # print("Source data balancing......")
    # xs.shape = (病例数，切片数，宽，高)

    # # 二分类版
    xs, ys1, ys2, ys3, s_order, s_height, s_width = return_dataset_train_2(source_dir, label_s_dir, modality=modality, slice_size=slice_size)
    xt, yt1, yt2, yt3, t_order, t_height, t_width = return_dataset_train_2(target_dir, label_t_dir, modality=modality, slice_size=slice_size)

    # 五分类版
    # xs, ys1, ys2, ys3, s_order, s_height, s_width = return_dataset_train(source_dir, label_s_dir, modality=modality,
    #                                                                        slice_size=slice_size)
    # xt, yt1, yt2, yt3, t_order, t_height, t_width = return_dataset_train(target_dir, label_t_dir, modality=modality,
    #                                                                        slice_size=slice_size)


    if mode == 0:  # 对纤维化进行采样
        xs_ROS, ys_ROS, xt_ROS, yt_ROS = xs, ys1, xt, yt1
    if mode == 1:  # 对脂变进行采样
        xs_ROS, ys_ROS, xt_ROS, yt_ROS = xs, ys2, xt, yt2
    if mode == 2:  # 对炎症进行采样
        xs_ROS, ys_ROS, xt_ROS, yt_ROS = xs, ys3, xt, yt3

    xs_sample_num = xs_ROS.shape[0]
    xt_sample_num = xt_ROS.shape[0]

    # 判断源域和目标域的尺寸是否相同（order: 切片张数）
    # if (s_order == t_order) & (s_height == t_height) & (s_width == t_width) & (xs_sample_num == xt_sample_num):
    #     order = s_order
    #     height = s_height
    #     width = s_width
    #     sample_num = xs_sample_num
    # else:
    #     print("The shape is not identical")

    order = s_order
    height = s_height
    width = s_width
    xs_ROS = xs_ROS.reshape((xs_sample_num, order, height, width))  # 重新调整尺寸
    xt_ROS = xt_ROS.reshape((xt_sample_num, order, height, width))

    return xs_ROS, ys_ROS, xt_ROS, yt_ROS

def generate_dataset(xs, ys, xt, yt, batch_size, gpu):
    S = {}
    T = {}

    S['imgs'] = xs.astype(np.float32)
    S['label'] = ys.astype(np.int32)
    T['imgs'] = xt.astype(np.float32)
    T['label'] = yt.astype(np.int32)

    data_loader = UnalignedDataLoader()  # 新建一个UnalignedDataLoader类
    data_loader.initialize(S, T, batch_size, batch_size, gpu)  # 将数据加载到gpu
    dataset = data_loader.load_data()

    return dataset

def generate_dataset_test(xs, ys, xt, yt, batch_size, gpu):
    S = {}
    T = {}

    S['imgs'] = xs.astype(np.float32)
    S['label'] = ys.astype(np.int32)
    T['imgs'] = xt.astype(np.float32)
    T['label'] = yt.astype(np.int32)

    data_loader = UnalignedDataLoader()  # 新建一个UnalignedDataLoader类
    data_loader.initialize(S, T, batch_size, batch_size, gpu)  # 将数据加载到gpu
    dataset = data_loader.load_data()
    # 最后 dataset = {'S': A, 'S_label': A_paths, 'T': B, 'T_label': B_paths} 生成一个可迭代对象

    return dataset

