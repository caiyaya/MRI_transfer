## 各个文件夹内存储的内容
attention_model：我的特征提取器ecanet20存储位置
configs：保存各个超参的文件
data：训练数据存储位置
data_analysis：用于各种数据可视化呈现
img：loss和acc训练趋势变化图的存储位置
loss：存放各种对齐损失及对抗损失
model_log：存放训练好后的网络参数
solver：训练主过程
static_dataloader：存放数据加载过程所需的过程
text_log：存放运行过程中控制台的输出内容作为记录


## 运行前要调整的内容：
1. configs/number_config.json  中的参数
其中可能会涉及到修改的是：
feature_dim：特征向量的输出维度
alpha，beta：损失函数的比例系数
epoch_num：epoch轮数
batch_size：顾名思义，我一般设置为16
optimizer：可选项有momentum，adam
scheduler：可选项有step，multi_step
learning_rate：optimizer中的学习率
momentum：optimizer中的动量值
weight_decay：optimizer中的权重衰减值
early_stop_step：当验证最高的准确率在early_stop_step个epoch后不再发生变化，则停止

2. 各项存储路径：
人和大鼠的数据存储路径（已经改成相对路径）
模型存储路径
loss与acc变化趋势图存储路径

## 运行方式：
直接运行main.py



P.S. 
1. 后续做迁移实验需要解开注释的地方：域对抗分类器部分和对齐损失部分
2. 由于我会把每次验证集中表现最好的都存起来，所以每次做完一次实验后model_log中的文件就要清空一次，不然有可能会加载到之前保存的模型

如果您在运行过程中有什么问题欢迎随时联系我~
