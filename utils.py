import torch
import torch.nn as nn
import itertools
from tqdm import tqdm
import numpy as np
import sys
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from matplotlib import pyplot as plt
import os
import shutil
from sklearn import svm
from itertools import combinations


def data_group(x, y, gpu):
    # if gpu:
    #     for cls_num in range(class_num):
    #         exec("x_c%s = torch.Tensor().cuda()" % cls_num)
    # else:
    #     for cls_num in range(class_num):
    #         exec("x_c%s = torch.Tensor()" % cls_num)
    #
    # data = x
    # label = y
    # list_x_c = ()
    #
    # for cls_num in range(class_num):
    #     for i in range(data.shape[0]):
    #         if (label[i] == cls_num):
    #             exec("x_c%s = torch.cat((x_c%s, torch.reshape(data[i], (1,-1))), 0)" % (cls_num, cls_num))
    #         exec("list_x_c.append())
    # if(gpu):
    #     x_c0 = torch.Tensor().cuda()
    #     x_c1 = torch.Tensor().cuda()
    #     x_c2 = torch.Tensor().cuda()
    # else:
    #     x_c0 = torch.Tensor()
    #     x_c1 = torch.Tensor()
    #     x_c2 = torch.Tensor()
    #
    # data = x
    # label = y
    #
    # for i in range(data.shape[0]):
    #     if (label[i] == 0):
    #         x_c0 = torch.cat((x_c0, torch.reshape(data[i], (1,-1))), 0)
    #     if (label[i] == 1):
    #         x_c1 = torch.cat((x_c1, torch.reshape(data[i], (1,-1))), 0)
    #     if (label[i] == 2):
    #         x_c2 = torch.cat((x_c2, torch.reshape(data[i], (1,-1))), 0)
    # return x_c0, x_c1, x_c2

    # 💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩 shit code 1 💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩
    if gpu :
        x_c0 = torch.Tensor().cuda()
        x_c1 = torch.Tensor().cuda()
        x_c2 = torch.Tensor().cuda()
        x_c3 = torch.Tensor().cuda()
        x_c4 = torch.Tensor().cuda()
        x_c5 = torch.Tensor().cuda()
        x_c6 = torch.Tensor().cuda()
        x_c7 = torch.Tensor().cuda()
        x_c8 = torch.Tensor().cuda()
        x_c9 = torch.Tensor().cuda()
    else:
        x_c0 = torch.Tensor()
        x_c1 = torch.Tensor()
        x_c2 = torch.Tensor()
        x_c3 = torch.Tensor()
        x_c4 = torch.Tensor()
        x_c5 = torch.Tensor()
        x_c6 = torch.Tensor()
        x_c7 = torch.Tensor()
        x_c8 = torch.Tensor()
        x_c9 = torch.Tensor()

    data = x
    label = y

    for i in range(data.shape[0]):
        if (label[i] == 0):
            x_c0 = torch.cat((x_c0, torch.reshape(data[i], (1,-1))), 0)
        if (label[i] == 1):
            x_c1 = torch.cat((x_c1, torch.reshape(data[i], (1,-1))), 0)
        if (label[i] == 2):
            x_c2 = torch.cat((x_c2, torch.reshape(data[i], (1,-1))), 0)
        if (label[i] == 3):
            x_c3 = torch.cat((x_c3, torch.reshape(data[i], (1,-1))), 0)
        if (label[i] == 4):
            x_c4 = torch.cat((x_c4, torch.reshape(data[i], (1,-1))), 0)
        if (label[i] == 5):
            x_c5 = torch.cat((x_c5, torch.reshape(data[i], (1,-1))), 0)
        if (label[i] == 6):
            x_c6 = torch.cat((x_c6, torch.reshape(data[i], (1,-1))), 0)
        if (label[i] == 7):
            x_c7 = torch.cat((x_c7, torch.reshape(data[i], (1,-1))), 0)
        if (label[i] == 8):
            x_c8 = torch.cat((x_c8, torch.reshape(data[i], (1,-1))), 0)
        if (label[i] == 9):
            x_c9 = torch.cat((x_c9, torch.reshape(data[i], (1,-1))), 0)

    return x_c0, x_c1, x_c2, x_c3, x_c4, x_c5, x_c6, x_c7, x_c8, x_c9

# 💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩 shit code 5 💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩
def classes_mean(data0, data1, data2, data3, data4, data5, data6, data7, data8, data9, gpu):
    # if(gpu):
    #     class_mean = torch.zeros((3, data0.shape[1])).cuda()
    # else:
    #     class_mean = torch.zeros((3, data0.shape[1]))
    # class_mean[0] = torch.mean(data0, dim=0, keepdim=True)
    # class_mean[1] = torch.mean(data1, dim=0, keepdim=True)
    # class_mean[2] = torch.mean(data2, dim=0, keepdim=True)

    if(gpu):
        class_mean = torch.zeros((10, data0.shape[1])).cuda()
    else:
        class_mean = torch.zeros((10, data0.shape[1]))
    class_mean[0] = torch.mean(data0, dim=0, keepdim=True)
    class_mean[1] = torch.mean(data1, dim=0, keepdim=True)
    class_mean[2] = torch.mean(data2, dim=0, keepdim=True)
    class_mean[3] = torch.mean(data3, dim=0, keepdim=True)
    class_mean[4] = torch.mean(data4, dim=0, keepdim=True)
    class_mean[5] = torch.mean(data5, dim=0, keepdim=True)
    class_mean[6] = torch.mean(data6, dim=0, keepdim=True)
    class_mean[7] = torch.mean(data7, dim=0, keepdim=True)
    class_mean[8] = torch.mean(data8, dim=0, keepdim=True)
    class_mean[9] = torch.mean(data9, dim=0, keepdim=True)

    return class_mean


def maximization_matrix(A, B, gpu):
    A_row_num = A.shape[0]
    B_row_num = B.shape[0]
    A_column_num = A.shape[1]
    B_column_num = B.shape[1]

    if(A_row_num==B_row_num)&(A_column_num==B_column_num):
        row_num = A_row_num
        column_num = A_column_num
    else:
        print("The row number and the column number is not identical.")
        sys.exit()

    if(gpu):
        max_matrix = torch.zeros((row_num, column_num)).cuda()
    else:
        max_matrix = torch.zeros((row_num, column_num))

    for r in range(row_num):
        max_matrix[r] = torch.max(A[r], B[r])

    return max_matrix


def plot_confusion_matrix(cm, classes,
                          save_tag = '',
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./img/' + save_tag + '_cfm.png')
    plt.close('all')  # 关闭图


def accuracy(x_out, y_true, classes, isPlot, save_tag=''):
    x_softmax = nn.Softmax(dim=1)  # 沿维度1进行softmax操作
    x_pro = x_softmax(x_out)
    y_pred = torch.argmax(x_pro, dim=1)

    # 计算混淆矩阵
    y = np.zeros(len(y_true))
    y_ = np.zeros(len(y_true))
    for i in range(len(y_true)):
        y[i] = y_true[i]
        y_[i] = y_pred[i]
    cnf_mat = confusion_matrix(y, y_)
    # print('cnf_mat = ', cnf_mat)

    if classes > 2:
        if isPlot:
            # # 绘制混淆矩阵
            plot_confusion_matrix(cnf_mat, range(classes), save_tag=save_tag)
            # 计算多分类评价值

    Acc = accuracy_score(y, y_, normalize=True)
    # print("Acc = ", Acc)
    # Sens = recall_score(y, y_, average='macro')
    # Prec = precision_score(y, y_, average='macro')
    # F1 = f1_score(y, y_, average='weighted')
    return Acc # , Sens, Prec, F1 #, cnf_mat

def accuracy1(x_svm, x_out, y_true, classes, isPlot, save_tag=''):
    x_softmax = nn.Softmax(dim=1)  # 沿维度1进行softmax操作
    x_pro = x_softmax(x_out)
    x_pro = (x_svm + x_pro) / 2
    y_pred = torch.argmax(x_pro, dim=1)
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    # 这里增加 svm的预测？

    # 计算混淆矩阵
    y = np.zeros(len(y_true))
    y_ = np.zeros(len(y_true))
    for i in range(len(y_true)):
        y[i] = y_true[i]
        y_[i] = y_pred[i]
    cnf_mat = confusion_matrix(y, y_)
    # print('cnf_mat = ', cnf_mat)

    # 记录错分样本
    error_list = []
    for i in range(len(y_true)):
        true = y_true[i].item()
        pred = y_pred[i].item()
        if true != pred:
            error_list.append(true)
    print("Error Classification:", error_list)

    if classes > 2:
        if isPlot:
            # # 绘制混淆矩阵
            plot_confusion_matrix(cnf_mat, range(classes), save_tag=save_tag)
            # 计算多分类评价值

    Acc = accuracy_score(y, y_, normalize=True)
    # print("Acc = ", Acc)
    # Sens = recall_score(y, y_, average='macro')
    # Prec = precision_score(y, y_, average='macro')
    # F1 = f1_score(y, y_, average='weighted')
    return Acc # , Sens, Prec, F1 #, cnf_mat

def xt_accuracy(y_pred, y_true, classes, isPlot, save_tag=''):
    # 计算混淆矩阵
    y = np.zeros(len(y_true))
    y_ = np.zeros(len(y_true))
    for i in range(len(y_true)):
        y[i] = y_true[i]
        y_[i] = y_pred[i]
    cnf_mat = confusion_matrix(y, y_)

    if classes > 2:
        if isPlot:
            # # 绘制混淆矩阵
            plot_confusion_matrix(cnf_mat, range(classes), save_tag=save_tag)
            # 计算多分类评价值
        Acc = accuracy_score(y, y_, normalize='True')
        Sens = recall_score(y, y_, average='macro')
        Prec = precision_score(y, y_, average='macro')
        F1 = f1_score(y, y_, average='weighted')
        return Acc, Sens, Prec, F1 #cnf_mat

# def synthesize(Acc, Loss, Sens, Prec, F1):
def synthesize(Acc, Loss):
    """
    计算平均 acc等参数
    """
    Acc_all = 0
    Loss_all = 0
    # Sens_all = 0
    # Prec_all = 0
    # F1_all = 0

    # print("ACC TUPLE= ", Acc)

    for num in Loss:
        Loss_all = Loss_all + num
    Loss_last = Loss_all / len(Loss)

    for num in Acc:
        Acc_all = Acc_all + num
    Acc_last = Acc_all / len(Acc)

    # for num in Loss:
    #     Loss_all = Loss_all + num
    # Loss_last = Loss_all / len(Loss)

    # for num in Sens:
    #     Sens_all = Sens_all + num
    # Sens_last = Sens_all / len(Sens)
    #
    # for num in Prec:
    #     Prec_all = Prec_all + num
    # Prec_last = Prec_all / len(Prec)
    #
    # for num in F1:
    #     F1_all = F1_all + num
    # F1_last = F1_all / len(F1)

    # Cnf_mat_new = Cnf_mat[0]
    # for num in range(1, len(Cnf_mat)):
    #     Cnf_mat_new = Cnf_mat_new + Cnf_mat[num]

    return Acc_last, Loss_last #, Sens_last, Prec_last, F1_last, Cnf_mat_new.tolist()


def copy_allfiles(src, dest):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)


def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot


def compute_proxy_distance(source_X, target_X):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    """
    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')
    """

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source / 2), int(nb_target / 2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        """
        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))
        """

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)

"""
print('==================================================================')
print('Computing PAD on DANN representation...')
pad_daan = compute_proxy_distance(s_feature_array, t_feature_array)
print('PAD on DANN representation = %f' % pad_daan)

print('Computing PAD on original data...')
pad_original = compute_proxy_distance(xs, xt)
print('PAD on original data = %f' % pad_original)
"""


# ----------------------------------三元组损失所需组件--------------------------- #
def pdist(vectors):
    """
    求一个矩阵中向量之间的距离
    例如：vectors = tensor([[1 2 3]
                           [2 2 2]
                           [2 2 2]])
        则：distance_matrix = tensor([[0., 2., 2.],
                                     [2., 0., 0.],
                                     [2., 0., 0.]])

    :param vectors: tensor类别的矩阵
    :return: tensor类别的距离矩阵
    """
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    # torch.mm(a,b) or a.mm(self,b) 矩阵a和b相乘
    return distance_matrix


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class TripletSelector(object):
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError  # 表示后续子类在继承这个父类时必须要重写这个方法


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        # print("全部待选label = ", labels)
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)  # 得到距离矩阵
        # print("距离矩阵为 = ", distance_matrix)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            # print("当前label为 = ", label)
            label_mask = (labels == label)  # 得到的是true和false的序列
            label_indices = np.where(label_mask)[0]   # 返回np中和label_mask的值相同的行的序号
            label_indices = list(label_indices)
            # print("label_indices = ", label_indices)
            if len(label_indices) < 2:
                continue
            # print("anchor_positives = ", list(combinations(label_indices, 2)))
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            negative_indices = np.where(np.logical_not(label_mask))[0]
            if len(negative_indices) == 0:
                continue

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                """
                if all(anchor_positive == np.array([0,1])):
                    continue
                """
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

            if len(triplets) == 0:
                triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)
# ----------------------------------------三元组所需组件完----------------------------------------------- #






# 下面都是kmeans的代码
# ToDo: Can't choose a cluster if two points are too close to each other, that's where the nan come from
# def kmeans_initialize(X, num_clusters):
#     """
#     initialize cluster centers
#     :param X: (torch.tensor) matrix
#     :param num_clusters: (int) number of clusters
#     :return: (np.array) initial state
#     """
#     num_samples = len(X)
#     indices = np.random.choice(num_samples, num_clusters, replace=False)
#     initial_state = X[indices]
#     return initial_state
#
#
# def kmeans(
#         X,
#         num_clusters,
#         gpu,
#         distance='euclidean',
#         cluster_centers=[],
#         tol=1e-4,
#         tqdm_flag=False,
#         iter_limit=0,
# ):
#     """
#     perform kmeans
#     :param X: (torch.tensor) matrix
#     :param num_clusters: (int) number of clusters
#     :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
#     :param tol: (float) threshold [default: 0.0001]
#     :param tqdm_flag: Allows to turn logs on and off
#     :param iter_limit: hard limit for max number of iterations
#     :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
#     """
#
#     if distance == 'euclidean':
#         pairwise_distance_function = pairwise_distance
#     elif distance == 'cosine':
#         pairwise_distance_function = pairwise_cosine
#     else:
#         raise NotImplementedError
#
#     # convert to float
#     X = X.float()
#
#     # transfer to device
#     if gpu:
#         X = X.cuda()
#     else:
#         X = X
#
#     # initialize
#     if type(cluster_centers) == list:  # ToDo: make this less annoyingly weird
#         initial_state = kmeans_initialize(X, num_clusters)
#     else:
#         print('resuming')
#         # find data point closest to the initial cluster center
#         initial_state = cluster_centers
#         dis = pairwise_distance_function(X, initial_state, gpu)
#         choice_points = torch.argmin(dis, dim=0)
#         initial_state = X[choice_points]
#         if gpu:
#             initial_state = initial_state.cuda()
#         else:
#             initial_state = initial_state
#
#
#     iteration = 0
#     if tqdm_flag:
#         tqdm_meter = tqdm(desc='[running kmeans]')
#     while True:
#
#         dis = pairwise_distance_function(X, initial_state, gpu)
#
#         choice_cluster = torch.argmin(dis, dim=1)
#
#         initial_state_pre = initial_state.clone()
#
#         for index in range(num_clusters):
#             if gpu:
#                 selected = torch.nonzero(choice_cluster == index).squeeze().cuda()
#             else:
#                 selected = torch.nonzero(choice_cluster == index).squeeze()
#
#             selected = torch.index_select(X, 0, selected)
#
#             initial_state[index] = selected.mean(dim=0)
#
#         center_shift = torch.sum(
#             torch.sqrt(
#                 torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
#             ))
#
#         # increment iteration
#         iteration = iteration + 1
#
#         # update tqdm meter
#         if tqdm_flag:
#             tqdm_meter.set_postfix(
#                 iteration=f'{iteration}',
#                 center_shift=f'{center_shift ** 2:0.6f}',
#                 tol=f'{tol:0.6f}'
#             )
#             tqdm_meter.update()
#         if center_shift ** 2 < tol:
#             break
#         if iter_limit != 0 and iteration >= iter_limit:
#             break
#
#     return choice_cluster, initial_state
#
#
# def kmeans_predict(
#         X,
#         cluster_centers,
#         gpu,
#         distance='euclidean',
# ):
#     """
#     predict using cluster centers
#     :param X: (torch.tensor) matrix
#     :param cluster_centers: (torch.tensor) cluster centers
#     :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
#     :return: (torch.tensor) cluster ids
#     """
#
#     if distance == 'euclidean':
#         pairwise_distance_function = pairwise_distance
#     elif distance == 'cosine':
#         pairwise_distance_function = pairwise_cosine
#     else:
#         raise NotImplementedError
#
#     # convert to float
#     X = X.float()
#
#     # transfer to device
#     if gpu:
#         X = X.cuda()
#     else:
#         X = X
#
#
#     dis = pairwise_distance_function(X, cluster_centers, gpu)
#     choice_cluster = torch.argmin(dis, dim=1)
#
#     return choice_cluster.cpu()
#
#
# def pairwise_distance(data1, data2, gpu):
#     # transfer to device
#     if gpu:
#         data1, data2 = data1.cuda(), data2.cuda()
#     else:
#         data1, data2 = data1, data2
#
#     # N*1*M
#     A = data1.unsqueeze(dim=1)
#
#     # 1*N*M
#     B = data2.unsqueeze(dim=0)
#
#     eps = 1e-6
#     dis = (A - B) ** 2.0 + eps
#
#     # return N*N matrix for pairwise distance
#     dis = dis.sum(dim=-1).squeeze()
#     return dis
#
#
# def pairwise_cosine(data1, data2, gpu):
#     # transfer to device
#     if gpu:
#         data1, data2 = data1.cuda(), data2.cuda()
#     else:
#         data1, data2 = data1, data2
#
#     # N*1*M
#     A = data1.unsqueeze(dim=1)
#
#     # 1*N*M
#     B = data2.unsqueeze(dim=0)
#
#     # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
#     A_normalized = A / A.norm(dim=-1, keepdim=True)
#     B_normalized = B / B.norm(dim=-1, keepdim=True)
#
#     cosine = A_normalized * B_normalized
#
#     # return N*N matrix for pairwise distance
#     cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
#     return cosine_dis
#
# class KMeans:
#     def __init__(self, n_clusers, max_iter = None, verbose=True, gpu=False):
#         self.n_clusters = n_clusers
#         self.labels = None
#         self.dists = None
#         self.centers = None
#         self.variation = torch.Tensor([float("Inf")]).cuda()
#         self.verbose = verbose
#         self.started = False
#         self.representative_samples = None
#         self.max_iter = max_iter
#         self.count = 0
#         self.gpu = gpu
#
#     def fit(self, x):
#         # 随机选择初始中心点：返回均匀分布的[0,x.shape[0]]之间的整数随机值
#         if(self.gpu):
#             init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).cuda()
#         else:
#             init_row = torch.randint(0, x.shape[0], (self.n_clusters,))
#         init_points = x[init_row]
#         self.centers = init_points
#
#
#         while True:
#             # 聚类标记
#             self.nearest_center(x)
#             # 更新中心点
#             self.update_center(x)
#             if self.verbose:
#                 print(self.variation, torch.argmin(self.dists, (0)))
#             if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
#                 break
#             elif self.max_iter is not None and self.count == self.max_iter:
#                 break
#
#             self.count += 1
#
#         return self.centers
#
#     def nearest_center(self, x):
#         if(self.gpu):
#             labels = torch.empty((x.shape[0],)).long().cuda()
#             dists = torch.empty((0, self.n_clusters)).cuda()
#         else:
#             labels = torch.empty((x.shape[0],)).long()
#             dists = torch.empty((0, self.n_clusters))
#
#         for i, sample in enumerate(x):
#             eps = 1e-6
#             dist = torch.sum(torch.mul(sample-self.centers, sample-self.centers), (1)) + eps
#             labels[i] = torch.argmin(dist)
#             dists = torch.cat([dists, dist.unsqueeze(0)], (0))
#         self.labels = labels
#         if self.started:
#             self.variation = torch.sum(self.dists-dists)
#         self.dists = dists
#         self.started = True
#
#     def update_center(self, x):
#         if(self.gpu):
#             centers = torch.empty((0, x.shape[1])).cuda()
#         else:
#             centers = torch.empty((0, x.shape[1]))
#         for i in range(self.n_clusters):
#             mask = self.labels == i
#             cluster_samples = x[mask]
#             centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
#         self.centers = centers
#
#     def representative_sample(self):
#         # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
#         self.representative_samples = torch.argmin(self.dists, (0))
#

# def accuracy(y_true, y_pred, classes, isPlot, save_tag=''):
#     # 计算混淆矩阵
#     y = np.zeros(len(y_true))
#     y_ = np.zeros(len(y_true))
#     for i in range(len(y_true)):
#         # print y[i],"--------------------",y_true[i]
#         y[i] = y_true[i]
#         y_[i] = y_pred[i]
#     # print y,y_
#     cnf_mat = confusion_matrix(y, y_)
#     # print cnf_mat
#
#     if classes > 2:
#         if isPlot:
#             # # 绘制混淆矩阵
#             plot_confusion_matrix(cnf_mat, range(classes), save_tag=save_tag)
#             # 计算多分类评价值
#         Acc = accuracy_score(y, y_, normalize='True')
#         Sens = recall_score(y, y_, average='macro')
#         Prec = precision_score(y, y_, average='macro')
#         F1 = f1_score(y, y_, average='weighted')
#         Support = precision_recall_fscore_support(y, y_, beta=0.5, average=None)
#         # print Support
#         return Acc, Sens, Prec, F1, cnf_mat
#     else:
#         Acc = 1.0 * (cnf_mat[1][1] + cnf_mat[0][0]) / len(y_true)
#         Sens = 1.0 * cnf_mat[1][1] / (cnf_mat[1][1] + cnf_mat[1][0])
#         Spec = 1.0 * cnf_mat[0][0] / (cnf_mat[0][0] + cnf_mat[0][1])
#         if isPlot:
#             # # 绘制混淆矩阵
#             plot_confusion_matrix(cnf_mat, range(classes), save_tag=save_tag)
#             # # 绘制ROC曲线
#             fpr, tpr, thresholds = roc_curve(y_true[:, 1], y_pred[:, 1])
#             fpr[0], tpr[0] = 0, 0
#             fpr[-1], tpr[-1] = 1, 1
#
#             Auc = auc(fpr, tpr)
#             plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (Auc))
#
#             plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color=(0.6, 0.6, 0.6), alpha=.8)
#
#             plt.xlim([-0.05, 1.05])
#             plt.ylim([-0.05, 1.05])
#             plt.xlabel('False Positive Rate')
#             plt.ylabel('True Positive Rate')
#             plt.title('Receiver operating characteristic example')
#             plt.legend(loc="lower right")
#             plt.savefig('img/' + save_tag + '_roc.png')
#             # plt.show()
#             plt.close('all')  # 关闭图
#
#             # # 记录ROC曲线以及曲线下面积
#             f = open('img/roc_record.txt', 'ab+')
#             f.write(save_tag + 'AUC:' + str(Auc) + '\n')
#             f.write('FPR:' + str(list(fpr)) + '\n')
#             f.write('TPR:' + str(list(tpr)) + '\n\n')
#             f.close()
#
#             # #字典中的key值即为csv中列名
#             # dataframe = pd.DataFrame({'FPR':fpr,'TPR':tpr})
#             # #将DataFrame存储为csv,index表示是否显示行名，default=True
#             # dataframe.to_csv('img/roc_record.csv', index=False, sep=',')
#
#         # 计算AUC值
#         Auc = roc_auc_score(y_true[:, 1], y_pred[:, 1])
#         return Acc, Sens, Spec, Auc
