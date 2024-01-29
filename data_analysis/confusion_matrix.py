import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools


# 绘制混淆矩阵
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
    # plt.close('all')  # 关闭图


if __name__ == '__main__':
    y_true = np.array(
        [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0]
    )
    y_score = np.array([
        .9, .8, .7, .6, .55, .54, .53, .52, .51, .505,
        .4, .39, .38, .37, .36, .35, .34, .33, .3, .1
    ])
    y_pred = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )

    # 混淆矩阵存储位置及图名
    save_path = r""
    save_tag = ''

    cm = confusion_matrix(y_true, y_pred)  # 输入为整型的真值和预测结果

    # 绘制混淆矩阵热图
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig('./img/' + save_tag + '_cfm.png')
    plt.show()
