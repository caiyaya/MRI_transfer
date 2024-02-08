import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# 读取Excel文件
file_path = 'parament.xlsx'
df = pd.read_excel(file_path, sheet_name='class2', engine='openpyxl')

# 选取第二列到第七列作为数据data，第八列作为标签label
# 假设列是按照从0开始计数的，第二列的索引是1，第七列的索引是6，第八列的索引是7
data = df.iloc[:, 1:7]
label = df.iloc[:, 7]

# 显示数据和标签的前几行，以验证读取是否正确
print("数据（data）的前几行:")
print(data.head())
print("\n标签（label）的前几行:")
print(label.head())

# 初始化K折交叉验证，这里使用5折
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化SVM分类器
svm_classifier = SVC(kernel='rbf')
# 用于存储每一折的准确率
acc_scores = []

for train_index, test_index in kf.split(data):
    # 分割数据集
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = label[train_index], label[test_index]


    # 训练SVM分类器
    svm_classifier.fit(X_train, y_train)

    # 预测测试集
    y_pred = svm_classifier.predict(X_test)

    # 计算并存储准确率
    acc = accuracy_score(y_test, y_pred)

    print(f"真值为：{y_test}")
    print(f"预测结果为：{y_pred}")
    print(f"Accuracy: {acc}")
