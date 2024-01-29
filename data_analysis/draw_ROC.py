import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# 仅支持二分类
def roc_auc(y_true, y_score):
    """
    y_true：真实标签
    y_score：模型预测分数(即softmax后的结果)
    pos_label：正样本标签，如“1”
    """
    # ------------ 以下是实现原理 ------------ #
    # 统计正样本和负样本的个数
    # num_positive_examples = (y_true == pos_label).sum()
    # num_negtive_examples = len(y_true) - num_positive_examples
    #
    # tp, fp = 0, 0
    # tpr, fpr, thresholds = [], [], []
    # score = max(y_score) + 1
    #
    # # 根据排序后的预测分数分别计算fpr和tpr
    # for i in np.flip(np.argsort(y_score)):
    #     # 处理样本预测分数相同的情况
    #     if y_score[i] != score:
    #         fpr.append(fp / num_negtive_examples)
    #         tpr.append(tp / num_positive_examples)
    #         thresholds.append(score)
    #         score = y_score[i]
    #
    #     if y_true[i] == pos_label:
    #         tp += 1
    #     else:
    #         fp += 1
    #
    # fpr.append(fp / num_negtive_examples)
    # tpr.append(tp / num_positive_examples)
    # thresholds.append(score)
    #
    # return fpr, tpr, thresholds
    # ------------ 实现原理 ------------ #

    # ------------ 实际上实现 ----------- #
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


# 绘制roc图
def draw_roc(fpr, tpr, roc_auc):
    plt.plot(fpr, tpr,color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.axis("square")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.show()

    print('AUC:', roc_auc)


y_true = np.array(
    [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
)
y_score = np.array([
    .9, .8, .7, .6, .55, .54, .53, .52, .51, .505,
    .4, .39, .38, .37, .36, .35, .34, .33, .3, .1
])

fpr, tpr, auc = roc_auc(y_true, y_score)
draw_roc(fpr, tpr, auc)

