import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# # 生成示例数据
np.random.seed(42)
num_samples = 200
num_features = 128

# 生成特征向量
features = np.random.rand(num_samples, num_features)

# 生成类别标签（假设有两类，0和1）
labels = np.random.randint(2, size=num_samples)
#
# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
features_low_dimensional = tsne.fit_transform(features)

# 绘制可视化图
plt.figure(figsize=(8, 6))
scatter = plt.scatter(features_low_dimensional[:, 0], features_low_dimensional[:, 1], c=labels, cmap='viridis')
plt.title('t-SNE Visualization of Feature Vectors with Class Labels')
plt.colorbar(scatter, label='Class Label')
plt.show()

# 二维可视化结果图
# plt.figure(figsize=(8, 6))
# plt.scatter(X_low_dimensional[:, 0], X_low_dimensional[:, 1], c=['blue']*50 + ['red'])
# plt.title('t-SNE Visualization of Two Classes')
# plt.show()



# # 设置均值
# # mean = np.array([1, 3, 4])
# num_components = 128
# mean1 = np.random.uniform(0, 1, size=num_components)
# mean2 = np.random.normal(loc=1, scale=1, size=128)
#
# # 生成一组向量
# num_vectors = 100
# # 分类情况主要取决于均值，加上同一均值的分布就是会完全混一起
# vectors1 = np.random.rand(num_vectors, 128) + mean1
# vectors2 = np.random.rand(num_vectors, 128) + mean1
# vectors3 = np.random.rand(num_vectors, 128) + mean2
#
# X_high_dimensional = np.vstack([vectors1, vectors2, vectors3])
#
# tsne = TSNE(n_components=2, random_state=6)
# X_low_dimensional = tsne.fit_transform(X_high_dimensional)
#
#
# plt.figure(figsize=(8, 6))
# plt.scatter(X_low_dimensional[:, 0], X_low_dimensional[:, 1], c=['blue']*100 + ['red']*100 + ['yellow']*100)
# plt.title('t-SNE Visualization of Two Classes')
# plt.show()