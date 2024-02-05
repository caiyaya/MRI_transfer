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


## 0129 数据集 dataloader部分整理- 基础版的traintestSplit 划分
目前数据集重新组织格式如下：
- data
   - human
      - npy_128
      - label
   - mice
      - npy_128
      - label
mice作为源域，human作为目标域， 对human进行五折划分，其中0.8为train 0.1为valid 0.1为test，模型会根据在valid上的表现进行参数保存，最后在test部分进行模型性能评估

## 0131 dataloader 重构 增加 aug 和 5种不同的切片
// json中配置aug参数， 0 means 不使用增强数据；1 means 使用增强数据

## 0201 loss 函数改进，增加平方对抗损失函数 以及 类别敏感的正则化项

模型提优 建议：
1）实验参数调整：

  早停策略，根据loss 下降 去看一下大概需要多少epoch

  batch_size 的调整

  损失函数中的超参数

    修改后的对抗损失函数的正则化项

    加入三元组损失函数

    total loss的加参

    优化器的参数

  特征提取器网络提出的ok不ok 建议和参数规模相似的网络进行对比



2）数据集调整：（可以从最重要单元逐步向外扩展，挨个对比实验验证 到底新增数据集的引入是否有效，每次调整都需要关注损失函数的变化情况 防止过拟合 or 欠拟合）

  mri 图像需要哪几个切片？

  aug 增强是否需要？需要的话 需要哪种类型的增强？

  9中不同模态的信息，需要哪几个？

 增加 autoshell脚本
 运行方式 python AutoShell.py
 这里为了方便后台运行，终端输出的log 重定向到了log_new.log下 可以（后台运行程序）后 通过log日志看结果
 后台运行程序可以使用 nohup 或是 tmux 推荐使用 后者

## 0202 增加参数信息
1）修改dataloader 读取对应的参数信息
2）对齐数据维度
3）增加clf参数分类器分支（svm）
4) clf参数分类器预测结果 和 原深度学习分类器预测结果 融合（待测试 显卡又被占用了）

todo：（待补充）
1）优化对抗部分 调整dann架构 参照师姐的那份代码
2）autoshell 脚本修复 无法删除model log下日志
3）

## 0204
1）补充时间埋点信息：
基于aug的（只用了90 和 f 两种）：
    svm对于 train 部分的训练时间: 0.24130964279174805 秒
    aug 路径拼接运行时间: 0.018590927124023438 秒
    return_dataset_train_2数据加载运行时间: 48.57200074195862 秒
    class_num_list: {0.0: 4960, 1.0: 3400}
    class_num_list: {0.0: 660, 1.0: 405}
    各种reshape数据运行时间: 17.553617000579834 秒
    KflodDataloader数据加载运行时间: 66.14675784111023 秒
    clf valid 测试：
    clf valid Accuracy: 0.6666666666666666
    参数部分valid数据加载运行时间: 0.07518506050109863 秒
    载入dataloader数据运行时间: 6.160874843597412 秒
    Solver2准备时间: 72.85819864273071 秒
不用aug的：
    svm对于 train 部分的训练时间: 0.18814921379089355 秒
    aug 路径拼接运行时间: 0.003129243850708008 秒
    return_dataset_train_2数据加载运行时间: 7.381183624267578 秒
    class_num_list: {0.0: 620, 1.0: 425}
    class_num_list: {1.0: 135, 0.0: 220}
    各种reshape数据运行时间: 2.037931442260742 秒
    KflodDataloader数据加载运行时间: 9.42404818534851 秒
    clf valid 测试：
    clf valid Accuracy: 0.6444444444444445
    参数部分valid数据加载运行时间: 0.08872818946838379 秒
    载入dataloader数据运行时间: 0.8832740783691406 秒
    Solver2准备时间: 10.85409140586853 秒
补充：
    基于aug的：4mins*15*5 = 300 mins
        一个 epoch 中训练的时间: 234.85137271881104 秒
        一个 epoch 中valid的时间：0.21526050567626953
    不基于aug的：0.5*15*5  = 37 mins
        一个 epoch 中训练的时间: 28.75791597366333 秒
        一个 epoch 中valid的时间：0.21965742111206055

2）对抗损失函数权重beta 基于损失反馈的自适应调整策略
框架已搭建，已fix advNet 预测为0的问题；具体策略待尝试

3）参数优化调整、学习率、早停策略等
修改学习率为1e-3，milestone为 5 和 10 epoch，早停策略设定为15个epoch
valid acc loss 没有改变

4）fix dataloader 加载aug数据集错误问题
移除test中的数据增强；如需要测试增强，则需要改动整个dataloader方式，增加tta预测（代码改动较大）


todo：
5）dann框架中引入cdan
    已实现 代码见solver Kflod DCAN commit "cadn框架"（维度已对齐，但是目测没啥效果提升，要自适应调整参数）
6）dann layer 层增加 参数 控制 梯度回传 （参数需要调整，已实现回传调整框架）
    已实现 commit "使用梯度回传替代加权loss控制"
7）对齐策略
    已实现 cmd align loss 貌似量级很大；不确定是否有用
8) multi branch 多分支任务
    todo
9）模型集成
    todo

逻辑：
    ecanet 特征提取器
    1）svm传统的基于机器学习的分类器去做human的分类；
    2）结合ecanet 和 mice信息，基于迁移学习 给一个dann的网络
    3）1） + 2） 结合在一个模型框架当中 -- 》 xxxNet

    「创新点：」
            1）提出一个框架xxxNet
            2）对抗损失loss 平方的 + 正则项
            3）Net 本身的total_loss,  基于损失反馈的自适应调整策略
            4) 模型集成 （投票 加权之类的融合策略）

            5) 创新点？ -> dann 网络本身做一些创新？ 要有道理 有motivation

    -> multi branch 多任务学习？ 为啥引入迁移学习

