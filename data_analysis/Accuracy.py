import numpy as np


def calculate_metrics(true_labels, predicted_labels):
    # 计算准确率（Accuracy）
    accuracy = np.mean(true_labels == predicted_labels)

    # 计算混淆矩阵
    confusion_matrix = np.zeros((2, 2))
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label] += 1

    # 计算精确率（Precision）
    precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1]) if confusion_matrix[0, 1] + confusion_matrix[1, 1] != 0 else 0

    # 计算召回率（Recall）
    recall = confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1]) if confusion_matrix[1, 0] + confusion_matrix[1, 1] != 0 else 0

    # 计算 F1 值
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return accuracy, precision, recall, f1

# 示例使用
true_labels = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
predicted_labels = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1])

accuracy, precision, recall, f1 = calculate_metrics(true_labels, predicted_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)