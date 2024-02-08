import numpy as np
import os
import matplotlib.pyplot as plt
# import SimpleITK as sitk
import shutil

npy_path = r"./param.npy"
label_path = r"./label.npy"



data = np.load(npy_path)
# 训练数据归一化
# 按照列（模态）进行归一化
#    计算每列的范数
column_norms = np.linalg.norm(data, axis=0)
#    对每列进行归一化
normalized_data = data / column_norms

# 五分类
label = np.load(label_path)

# 二分类
label_2 = []
for i in label:
    if i < 3:
        label_2.append(0)
    else:
        label_2.append(1)
label_2 = np.array(label_2)

# ---------------------- 逻辑回归模型 ------------------- #
# max_acc = 0
# max_rand = 0
# avg_acc = 0
#
# # 观测随机种子对结果的影响
# for i in range(1, 60):
#     # 划分训练集和测试集
#     # 使用原始数据
#     # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=i)
#
#     # 归一化后的数据
#     X_train, X_test, y_train, y_test = train_test_split(normalized_data, label, test_size=0.3, random_state=i)
#     print(f"Random_state: {i}")
#
#     # 初始化逻辑回归模型
#     model = LogisticRegression()
#
#     # 训练模型
#     model.fit(X_train, y_train)
#
#     # 在测试集上进行预测
#     y_pred = model.predict(X_test)
#
#     # 计算准确率
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"模型在测试集上的准确率: {accuracy}")
#     avg_acc = avg_acc + accuracy
#
#     if accuracy > max_acc:
#         max_acc = accuracy
#         max_rand = i
#
# avg_acc = avg_acc / 60
#
# print(f"Max Accuracy: {max_acc}")
# print(f"对应random_state: {max_rand}")
# print(f"平均准确率: {avg_acc}")

# ---------------------- 逻辑回归模型 ------------------- #

# ---------------------- 多层感知机模型 ------------------- #
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# import torch.nn.functional as F
# # from torch.utils.tensorboard import SummaryWriter
#
# # 构建 Dataset 和 DataLoader
# numpy_array = np.array(data)
# X = torch.tensor(numpy_array, dtype=torch.double)
# # 按各个模态进行归一化
# # normalized_tensor = F.normalize(X, p=2, dim=0)
# y = torch.tensor(label_2)
#
#
# dataset = TensorDataset(X, y)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# train_dataset = TensorDataset(X_train, y_train)
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_dataset = TensorDataset(X_test, y_test)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
#
#
# # 定义简单的全连接神经网络模型
# class SimpleClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size).double()
#         self.fc2 = nn.Linear(hidden_size, hidden_size).double()
#         self.fc3 = nn.Linear(hidden_size, output_size).double()
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.softmax(x)
#         return x
#
#
# # 初始化模型、损失函数和优化器
# input_size = 6  # 特征维度
# hidden_size = 8  # 隐层维度
# output_size = 2  # 分类数
# model = SimpleClassifier(input_size, hidden_size, output_size)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#
# # 训练模型
# num_epochs = 100
# losses = []
# accuracies = []
# max_train_acc = 0
# max_test_acc = 0
#
# test_losses = []
# test_accuracies = []
# for epoch in range(num_epochs):
#     # 清零梯度
#     optimizer.zero_grad()
#     correct_count = 0
#     total_count = 0
#     epoch_loss = 0
#
#     for inputs, labels in train_dataloader:
#         # # 清零梯度
#         # optimizer.zero_grad()
#
#         # 前向传播
#         outputs = model(inputs)
#
#         # 计算损失
#         labels = labels.long()
#         loss = criterion(outputs, labels)
#         epoch_loss += loss.item()
#
#         # 反向传播和优化
#         loss.backward()
#         optimizer.step()
#
#         # 统计准确率
#         _, predicted = torch.max(outputs, 1)
#         correct_count += (predicted == labels).sum().item()
#         total_count += labels.size(0)
#
#     # 计算每个 epoch 的准确率
#     accuracy = correct_count / total_count
#     print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy}')
#     if max_train_acc < accuracy:
#         max_train_acc = accuracy
#
#     # 打印每个epoch的损失
#     avg_loss = epoch_loss / len(X)
#     print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')
#
#     # 将平均损失和准确率存储
#     losses.append(avg_loss)
#     accuracies.append(accuracy)
#
#     # 进行测试
#     model.eval()
#     correct_count = 0
#     total_count = 0
#     total_loss = 0.0
#     all_output = []
#
#     with torch.no_grad():
#         for inputs, targets in test_dataloader:
#             outputs = model(inputs)
#             targets = targets.long()
#             loss = criterion(outputs, targets)
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             correct_count += (predicted == targets).sum().item()
#             total_count += targets.size(0)
#         # 平均损失
#         avg_loss = total_loss / len(test_dataloader)
#         print(f'------Test_Loss: {avg_loss}')
#         # 平均准确率
#         accuracy = correct_count / total_count
#         print(f'------Test_Accuracy: {accuracy}')
#         if max_test_acc < accuracy:
#             max_test_acc = accuracy
#
#         # 将平均损失和准确率存储
#         test_losses.append(avg_loss)
#         test_accuracies.append(accuracy)
#
# print(f'=====Train_Accuracy_max : {max_train_acc}')
# print(f'=====Test_Accuracy_max : {max_test_acc}')
#
# model.eval()
# all_output = []
# all_true = []
# with torch.no_grad():
#     for inputs, targets in dataloader:
#         outputs = model(inputs)
#         outputs_npy = outputs.numpy()
#         all_output.append(outputs_npy)
#         all_true.append(targets)
#         targets = targets.long()
#         loss = criterion(outputs, targets)
#         total_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         correct_count += (predicted == targets).sum().item()
#         total_count += targets.size(0)
#     # 所有预测结果
#     all_feature = np.concatenate(all_output)
#     all_true = np.concatenate(all_true)


# # ---------- 绘制loss和acc的折线图 -------------- #
# plt.figure(figsize=(10, 5))
#
# plt.subplot(2, 2, 1)
# plt.plot(losses, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Over Time')
# plt.legend()
#
# plt.subplot(2, 2, 2)
# plt.plot(accuracies, label='Training Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy Over Time')
# plt.legend()
#
# plt.subplot(2, 2, 3)
# plt.plot(test_losses, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Test Loss Over Time')
# plt.legend()
#
# plt.subplot(2, 2, 4)
# plt.plot(test_accuracies, label='Test Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Test Accuracy Over Time')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
# ---------- 绘制loss和acc的折线图 -------------- #

    # # 测试
    # model.eval()  # 切换到评估模式
    # test_correct_count = 0
    # test_total_count = 0
    # test_loss = 0.0
    #
    # with torch.no_grad():
    #     for test_inputs, test_labels in dataloader:
    #         test_outputs = model(test_inputs)
    #         test_loss += criterion(test_outputs, test_labels).item()
    #
    #         _, test_predicted = torch.max(test_outputs, 1)
    #         test_correct_count += (test_predicted == test_labels).sum().item()
    #         test_total_count += test_labels.size(0)
    #
    # test_accuracy = test_correct_count / test_total_count
    # average_test_loss = test_loss / len(dataloader)
    #
    # print(f'Testing Accuracy: {test_accuracy:.4f}, Testing Loss: {average_test_loss:.4f}')

# ---------------------- 多层感知机模型 ------------------- #

# ---------------------- KNN -------------------------- #
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

max_acc = 0
max_rand = 0
avg_acc = 0
# 观测随机种子对结果的影响
for i in range(1, 60):
    # 划分训练集和测试集
    # 使用原始数据
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    avg_acc = 0  # 平均准确率
    max_acc = 0  # 最大准确率
    max_rand = None  # 得到最大准确率的随机状态
    for train_index, test_index in kf.split(data):
        # 分割数据集
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label_2[train_index], label_2[test_index]

        # 创建KNN分类器
        knn_classifier = KNeighborsClassifier(n_neighbors=4)

        # 训练模型
        knn_classifier.fit(X_train, y_train)

        # 预测测试集
        y_pred = knn_classifier.predict(X_test)

        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"真值为：{y_test}")
        print(f"预测结果为：{y_pred}")
        print(f"Accuracy: {accuracy}")

        avg_acc += accuracy

        if accuracy > max_acc:
            max_acc = accuracy
    # 计算并打印平均准确率
    avg_acc /= 5
    print(f"平均准确率为：{avg_acc}")
    print(f"最大准确率为：{max_acc}")

    # X_train, X_test, y_train, y_test = train_test_split(data, label_2, test_size=0.3, random_state=i)
    #
    # # 归一化后的数据
    # # X_train, X_test, y_train, y_test = train_test_split(normalized_data, label, test_size=0.3, random_state=i)
    # print(f"Random_state: {i}")
    #
    # # 创建KNN分类器
    # knn_classifier = KNeighborsClassifier(n_neighbors=4)
    #
    # # 训练模型
    # knn_classifier.fit(X_train, y_train)
    #
    # # 预测测试集
    # y_pred = knn_classifier.predict(X_test)
    #
    # print(f"真值为：{y_test}")
    # print(f"预测结果为：{y_pred}")
    #
    # # 计算准确率
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy}")

    avg_acc = avg_acc + accuracy

    if accuracy > max_acc:
        max_acc = accuracy
        max_rand = i

avg_acc = avg_acc / 60

print(f"Max Accuracy: {max_acc}")
print(f"对应random_state: {max_rand}")
print(f"平均准确率: {avg_acc}")


# ---------------------- K近邻 -------------------------- #

# ---------------------- 支持向量机 ----------------------- #
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score


# max_acc = 0
# max_rand = 0
# avg_acc = 0
# # 观测随机种子对结果的影响
# for i in range(1, 60):
#     # 划分训练集和测试集
#     # 原始数据
#     # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=i)
#     # 按列归一化数据
#     X_train, X_test, y_train, y_test = train_test_split(normalized_data, label, test_size=0.2, random_state=i)
#
#     # 创建SVM分类器
#     svm_classifier = SVC(kernel='linear', C=1.0)  # Accuracy: 0.7647058823529411
#     # svm_classifier = SVC(kernel='poly', degree=2, C=1.0)  # Accuracy: 0.7647058823529411  变化degree对结果没影响
#     # svm_classifier = SVC(kernel='rbf', C=1.0)  # Accuracy: 0.7647058823529411
#     # svm_classifier = SVC(kernel='sigmoid', C=1.0)  # Accuracy: 0.7058823529411765
#
#     # 训练模型
#     svm_classifier.fit(X_train, y_train)
#
#     # 预测测试集
#     y_pred = svm_classifier.predict(X_test)
#
#     # 计算准确率
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy}")
#     avg_acc = avg_acc + accuracy
#
#     if accuracy > max_acc:
#         max_acc = accuracy
#         max_rand = i
#
# avg_acc = avg_acc / 60
#
# print(f"Max Accuracy: {max_acc}")
# print(f"对应random_state: {max_rand}")
# print(f"平均准确率: {avg_acc}")

# ---------------------- 支持向量机 ----------------------- #