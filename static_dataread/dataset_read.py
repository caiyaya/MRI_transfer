from static_dataread.svhn import load_svhn
from static_dataread.mnist import load_mnist
from static_dataread.usps import load_usps
from static_dataread.unaligned_data_loader import UnalignedDataLoader
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
def clf_data_loader(extra, target, t_train_select):
    """
    准备训练数据和标签。

    参数:
    - extra: 包含特征数据文件的目录路径。
    - target: 包含标签数据文件的目录路径。
    - t_train_select: 选择的训练集ID列表。

    返回:
    - x_train: 训练数据数组。
    - y_train: 训练标签数组。
    """
    t_train_select_set = set(t_train_select)
    clf_train_list = []
    for file_path in os.listdir(extra):
        if file_path.split("_")[1].split(".")[0] in t_train_select_set:
            clf_train_list.append(os.path.join(extra, file_path))
    clf_train_label_list = []
    for file_path in clf_train_list:
        addr = file_path.replace('human_extra', 'human/label')
        clf_train_label_list.append(addr.split('.npy')[0])

    slices =  1
    # 准备分类器的数据
    x_train = []
    path_train = []
    y_train = []
    for data_path in clf_train_list:
        data = np.load(data_path)
        if slices == 1:
            x_train.append(data)
            path_train.append(data_path)
        else:
            for i in range(slices): ## 数据会扩增为五份
                x_train.append(data)
                path_train.append(data_path)


    for label_path in clf_train_label_list:
        if slices == 1:
            label_path = os.path.join(label_path + "_0", "label.txt")
            with open(label_path, 'r') as label_file:
                label = label_file.readline().strip().split(',')[0]  # 读取第一行的第一个值
                label = int(label)  # 转换为整数
                # 根据label值进行处理，小于3则为0，否则为1
                label = 0 if label < 3 else 1
                y_train.append(label)
        else:
            for index in range(slices):
                temp = label_path
                temp = temp + "_{}".format(index)
                temp = os.path.join(temp, "label.txt")
                with open(temp, 'r') as label_file:
                    label = label_file.readline().strip().split(',')[0]  # 读取第一行的第一个值
                    label = int(label)  # 转换为整数
                    # 根据label值进行处理，小于3则为0，否则为1
                    label = 0 if label < 3 else 1
                    y_train.append(label)


    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train, path_train


def return_dataset(domain_name, usps, scale, all_use):
    if domain_name == 'svhn':
        train_image, train_label, test_image, test_label = load_svhn()
    if domain_name == 'mnist':
        train_image, train_label, test_image, test_label = load_mnist(scale=scale,
                                                                      usps=usps,
                                                                      all_use=all_use)
    if domain_name == 'usps':
        train_image, train_label, test_image, test_label = load_usps(all_use=all_use)

    return train_image, train_label, test_image, test_label, domain_name


def read_dataset(source_name, target_name, scale=False, all_use='no'):
    usps = False
    if source_name == "usps" or target_name == "usps":
        usps = True

    xs_train, ys_train, xs_test, ys_test, s_domain_name = return_dataset(source_name,
                                                                         usps=usps,
                                                                         scale=scale,
                                                                         all_use=all_use)
    xt_train, yt_train, xt_test, yt_test, t_domain_name = return_dataset(target_name,
                                                                         usps=usps,
                                                                         scale=scale,
                                                                         all_use=all_use)

    return xs_train, ys_train, xt_train, yt_train, \
           xs_test, ys_test, xt_test, yt_test,\
           s_domain_name, t_domain_name


def generate_dataset(xs, ys, xt, yt, s_domain_name, t_domain_name, batch_size, gpu):
    S = {}
    T = {}

    S['imags'] = xs
    S['label'] = ys
    T['imags'] = xt
    T['label'] = yt

    # synth时，scale为40
    # usps时，scale为28
    # 其他时候，scale为32
    scale = 40 if s_domain_name == 'synth' else 28 if t_domain_name == 'usps' or t_domain_name == 'usps' else 32

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, gpu, scale=scale)
    dataset = train_loader.load_data()

    return dataset



