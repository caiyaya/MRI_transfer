from time import time

import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

# from readtxt import *
from sklearn.manifold import TSNE


source_name = 'electronics'
target_name = 'kitchen'

"""
Perplexity: 5-50. Be smaller than the number of points.
Epsilon: No fixed number of steps. Different data sets require different numbers of iterations to converge.
"""
# 单域五分类版
def get_data():
    s_data = np.loadtxt(r'/home/lab321/C/hx/code/HX_littlepaper/t_SNE/5/data.txt')
    s_label = np.loadtxt(r'/home/lab321/C/hx/code/HX_littlepaper/t_SNE/5/label.txt')
    data = s_data
    label = s_label
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    print("data.shape[0] / 2 = ", data.shape[0] / 2)
    for i in range(data.shape[0]):
        if label[i] == 1:
            ax.scatter(data[i, 0], data[i, 1], s=3, c='r', marker='o')
        elif label[i] == 2:
            ax.scatter(data[i, 0], data[i, 1], s=3, c='y', marker='o')
        elif label[i] == 3: # b
            ax.scatter(data[i, 0], data[i, 1], s=3, c='b', marker='o')
        elif label[i] == 4:
            ax.scatter(data[i, 0], data[i, 1], s=3, c='g', marker='o')
        elif label[i] == 5:
            ax.scatter(data[i, 0], data[i, 1], s=3, c='black', marker='o')
        else:
            ax.scatter(data[i, 0], data[i, 1], s=3, c='purple', marker='o')

    plt.xticks(np.arange(-0.1, 1.2, step=0.1))
    plt.yticks(np.arange(-0.1, 1.2, step=0.1))
    plt.title(title)



# 两域迁移二分类版
# def get_data():
#     s_data = np.loadtxt('./t_SNE/s_feature_' + source_name + target_name + 'None.txt')
#     t_data = np.loadtxt('./t_SNE/t_feature_' + source_name + target_name + 'None.txt')
#     s_label = np.loadtxt('./t_SNE/s_label_' + source_name + target_name + 'None.txt')
#     t_label = np.loadtxt('./t_SNE/t_label_' + source_name + target_name + 'None.txt')
#     data = np.concatenate((s_data, t_data), axis=0)
#     label = np.concatenate((s_label, t_label), axis=0)
#     n_samples, n_features = data.shape
#     return data, label, n_samples, n_features
#
#
# def plot_embedding(data, label, title):
#     x_min, x_max = np.min(data, 0), np.max(data, 0)
#     data = (data - x_min) / (x_max - x_min)
#
#     plt.figure()
#     ax = plt.subplot(111)
#     print(data.shape[0]/2)
#     for i in range(data.shape[0]):
#         if i < data.shape[0]/2:
#             if (label[i] == 1):
#                 ax.scatter(data[i, 0], data[i, 1], s = 3, c = 'r', marker = 'o') # source_pos
#             else:
#                 ax.scatter(data[i, 0], data[i, 1], s = 3, c = 'y', marker = 'o')  # source_neg
#         else:                                               # y
#             if (label[i] == 1):                             # b
#                 ax.scatter(data[i, 0], data[i, 1], s = 3, c = 'b', marker = 'o') # target_pos
#             else:
#                 ax.scatter(data[i, 0], data[i, 1], s = 3, c = 'g', marker = 'o') # target_neg
#
#     plt.xticks(np.arange(-0.1, 1.2, step=0.1))
#     plt.yticks(np.arange(-0.1, 1.2, step=0.1))
#     plt.title(title)


def main():
    data, label, n_samples, n_features = get_data()
    print("Computing t-SNE embedding")
    tsne = TSNE(n_components=2, perplexity=100, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    plot_embedding(result, label, ' ')
    plt.savefig(source_name + target_name + 'None.png')  # yishi


if __name__ == '__main__':
    main()
