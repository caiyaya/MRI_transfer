import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import math
from PIL import Image
import cv2
import json


# 把原始的数据加到增强的数据里面
# s_path = r"/home/lab321/C/hx/data/youyi/human/20240118/human_npy_origin"
# t_path = r"/home/lab321/C/hx/data/youyi/human/20240118/human_npy_chw"
# for stage in os.listdir(t_path):
#     tmp_read_1 = os.path.join(t_path, stage)
#     for data_class in os.listdir(tmp_read_1):
#         tmp_read_2 = os.path.join(tmp_read_1, data_class)
#         for npy in os.listdir(tmp_read_2):
#             s_npy = npy[:9] + ".npy"
#             tmp_s_path = os.path.join(s_path, s_npy)
#             tmp_t_path = os.path.join(tmp_read_2, s_npy)
#             shutil.copy(tmp_s_path, tmp_t_path)



# 构造list保存npy的读取路径和参数值以及对应标签，然后保存为json文件
# root_path = r"/home/lab321/C/hx/data/youyi/human/20240118"
# npy_path = "./npy"
# npy_read_path = "/home/lab321/C/hx/data/youyi/human/20240118/human_npy_chw"
# mean_read_path = r"/home/lab321/C/hx/data/youyi/human/20240118/human_mean"
# label_read_path = r"/home/lab321/C/hx/data/youyi/human/20240118/label"
#
#
# train_data_list = []
# valid_data_list = []
# test_data_list = []
# for stage in os.listdir(npy_read_path):
#     tmp_read_1 = os.path.join(npy_read_path, stage)
#     tmp_save_1 = os.path.join(npy_path, stage)
#     for data_class in os.listdir(tmp_read_1):
#         tmp_read_2 = os.path.join(tmp_read_1, data_class)
#         for npy in os.listdir(tmp_read_2):
#             tmp_read_3 = os.path.join(tmp_read_2, npy)
#             tmp_mean = os.path.join(mean_read_path, npy[:9] + ".npy")
#
#             mean = np.load(tmp_mean)
#             mean = mean.tolist()
#             label = data_class[1:2]
#
#             tmp_save_2 = os.path.join(tmp_save_1, npy)
#             elm = (tmp_save_2, mean, label)
#             if stage == "training":
#                 train_data_list.append(elm)
#             elif stage == "validation":
#                 valid_data_list.append(elm)
#             else:
#                 test_data_list.append(elm)
#
# json_path = '/home/lab321/C/hx/code/HX_littlepaper/hx_dataloader'
# train_json = os.path.join(json_path, 'train_list.json')
# valid_json = os.path.join(json_path, 'validation_list.json')
# test_json = os.path.join(json_path, 'test_list.json')
#
# with open(train_json, 'w') as json_file:
#     json.dump(train_data_list, json_file)
#
# with open(valid_json, 'w') as json_file:
#     json.dump(valid_data_list, json_file)
#
# with open(test_json, 'w') as json_file:
#     json.dump(test_data_list, json_file)









# # 初始文件路径
# root_path = r"/home/lab321/C/hx/data/youyi/human/20240118/using"
# read_path = r"/home/lab321/C/hx/data/youyi/human/20240118/human"
#
# # 最终生成的读取路径列表
# data_list = []
#
# data_type = ["training", "validation", "test"]
#
# for type in data_type:
#     # 定位保存路径
#     npy_path = os.path.join(root_path, type, "npy")
#     mean_path = os.path.join(root_path, type, "mean")
#     label_path = os.path.join(root_path, type, "label")
#
#     # 读取数据
#     read_npy_path = os.path.join(root_path, type)
#     read_mean_path = os.path.join(root_path, type)
#     read_label_path = os.path.join(root_path, type)
