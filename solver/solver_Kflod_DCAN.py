import os
import torch
import torch.optim as optim
from datetime import datetime
import time
from static_dataread.human_read import *
from model.transfer_model import *
from utils import *
from utils import accuracy
from loss.dis_loss import *
# from model.mogai_eca_resnet import *
from attention_model.ECAResNet import eca_resnet20
from attention_model.cbam_net import CBAM_Net
from loss.tripletloss import *
from loss import avd_loss
from static_dataread.dataset_read import clf_data_loader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from loss.cdan_loss import *


plt.switch_backend('Agg')

imgPath = "./caicaiImage/img"


class SolverSeed(object):
    def __init__(self,  x_train, y_train, mark, extra, s_train_select, t_path, t_train_select, t_valid_select,  model_save_path, config, clfModel):
        self.config = config
        self.mark = mark # 交叉验证具体哪一折
        self.mode = config.mode  # 判定是哪一个病症

        self.source = config.source  # 源域数据
        self.target = config.target  # 目标域数据

        self.class_num = config.class_num  # 类别数目（二分类 or 多分类）

        self.slice_size = config.slice_size
        self.modality = config.modality

        self.curr_epoch = 0
        self.feature_dim = config.feature_dim

        self.clf = clfModel  # 集成的机器学习分类器

        # 载入源数据、源标签、目标数据、目标标签
        print("dataset loading......")

        # 根据五折交叉验证 整理s和t相关的数据和标签， s_train_list 和 s_train_label_list; s_val_list 和 s_val_label_list
        s_train_list = s_train_select
        # 这里需要对aug增加特判逻辑
        s_train_label_list = []
        for s in s_train_list:
            if "aug" in s:
                parts = s.split('/')
                # 将'npy_aug'替换为'label'，并移除倒数第二个部分
                new_parts = parts[:-2] + [parts[-1]]
                new_parts[3] = 'label'
                new_path = '/'.join(new_parts)
                s_train_label_list.append(new_path)
            else:
                s_train_label_list.append(s.replace('npy', 'label'))


        # 这里要根据 t_train_select 构建 训练和测试数据对
        print("break point")
        t_train_select_set = set(t_train_select)
        t_train_list = []
        for file_path in os.listdir(t_path):
            if file_path.split('_')[1] in t_train_select_set and file_path.split('_')[2] == "0":
                t_train_list.append(os.path.join(t_path, file_path))

        t_train_label_list = [t.replace('npy', 'label') for t in t_train_list]

        t_valid_select_set = set(t_valid_select)
        t_valid_list = []
        for file_path in os.listdir(t_path):
            if file_path.split('_')[1] in t_valid_select_set and file_path.split('_')[2] == "0":
                t_valid_list.append(os.path.join(t_path, file_path))

        t_valid_label_list = [t.replace('npy', 'label') for t in t_valid_list]

        # 这里判断是否要增加aug部分的数据
        if config.aug == 1:
            t_train_list_aug = []
            t_path_aug = t_path.replace('npy', 'npy_aug')
            for file_path_augs in os.listdir(t_path_aug):
                for file_path in os.listdir(os.path.join(t_path_aug, file_path_augs)):
                    if file_path.split('_')[1] in t_train_select_set:
                        aug = os.path.join(file_path_augs, file_path)
                        t_train_list_aug.append(os.path.join(t_path_aug, aug))
            t_train_label_list_aug = []
            for t in t_train_list_aug:
                parts = t.split('/')
                # 将'npy_aug'替换为'label'，并移除倒数第二个部分
                new_parts = parts[:-2] + [parts[-1]]
                new_parts[3] = 'label'
                new_path = '/'.join(new_parts)
                t_train_label_list_aug.append(new_path)

            # 测试的时候valid 不做增强
            t_train_list.extend(t_train_list_aug)
            t_train_label_list.extend(t_train_label_list_aug)

        # 有oversampling的版本
        # oversampling 策略？
        self.xs_train, self.ys_train, self.xt_train, self.yt_train, self.xt_valid, self.yt_valid  = KflodDataloader(
            self.slice_size,
            self.modality, self.class_num,
            self.mode,
            source_dir=s_train_list,
            label_s_dir=s_train_label_list,
            target_dir=t_train_list,
            label_t_dir=t_train_label_list,
            target_v_dir=t_valid_list,
            label_tv_dir=t_valid_label_list,
        )

        # 增加逻辑 将valid部分拆分为 test 和 valid
        # self.xt_valid, self.yt_valid
        # 这里有问题，可能对某个human重复采样，测算其最后结果，尽可能评估不同病人的 -> 顺序采样即可
        # 这里应该直接前半部分是 valid 后半部分是test
        valid_size = len(self.xt_valid)

        indices = np.arange(valid_size)
        # np.random.shuffle(indices)
        split = int(valid_size * 0.5)  # 50%用于拆分
        self.xt_valid1 = np.array([self.xt_valid[i] for i in indices[:split]])
        self.yt_valid1 = np.array([self.yt_valid[i] for i in indices[:split]])
        self.xt_valid2 = np.array([self.xt_valid[i] for i in indices[split:]])
        self.yt_valid2 = np.array([self.yt_valid[i] for i in indices[split:]])

        x, y, path = clf_data_loader(extra, t_path, t_valid_select)
        splitline = split
        self.x_test = np.array([x[int(i)] for i in indices[:splitline]])
        self.y_test = np.array([y[int(i)] for i in indices[:splitline]])
        x_valid = [x[int(i)] for i in indices[splitline:]]
        y_valid = [y[int(i)] for i in indices[splitline:]]

        # print("clf valid 测试：")
        # for clf_name, clf_item in self.clf.items():
        #     y_pred_proba = clf_item.predict_proba(x_valid)
        #     # 选择概率最高的类别作为预测类别
        #     y_pred = y_pred_proba.argmax(axis=1)
        #
        #     # 打印每个分类器的名称（如果分类器没有名称属性，这里可能需要修改）
        #     print(f"\n分类器：{clf_name}")
        #     print("clf valid Accuracy:", accuracy_score(y_valid, y_pred))

        x = np.vstack((x_train, x_valid))
        y = np.concatenate((y_train, y_valid))
        for clf_name, clf_item in self.clf.items():
            clf_item.fit(x, y)
        # 分类器模型融合
        # 搜索逻辑 应该和 最终使用的时候的融合逻辑相同
        if config.searchSeed == 1:
            print("clf test 测试：")
            total_acc = 0.0
            number_clf = 0
            pred_proba = None
            for clf_name, clf_item in self.clf.items():
                number_clf += 1
                y_pred_proba = clf_item.predict_proba(self.x_test)
                if pred_proba is None:
                    pred_proba = np.zeros_like(y_pred_proba)
                pred_proba += y_pred_proba
                # 选择概率最高的类别作为预测类别
                y_pred = y_pred_proba.argmax(axis=1)
                acc = accuracy_score(self.y_test, y_pred)
                total_acc += acc
                # 打印每个分类器的名称
                print(f"\n分类器：{clf_name}")
                # 打印 y_pred 和 y_valid
                print("y_pred:", y_pred)
                print("y_test:", self.y_test)
                # print("y_prob:", y_pred_proba)
                print("clf test Accuracy:", acc)
            pred_proba = pred_proba / number_clf

            print("各分类器加权概率：{}".format(pred_proba))
            # 这里cfg 中配置可启动对应的阈值搜索 后期提点可用
            # 计算一个更好的分类的阈值
            # 尝试的阈值范围，例如从0.1到0.9，步长为0.01
            best_acc = 0.0
            best_thr = 0.5  # 默认阈值

            thr = self.config.thr
            if thr > 0.0:
                thresholds = np.arange(0.1, 0.9, 0.01)
                for thr in thresholds:
                    y_pred = (pred_proba[:, 1] > thr).astype(int)  # 根据阈值修改预测
                    avg_acc = accuracy_score(self.y_test, y_pred)  # 计算准确率
                    # 更新最佳准确率和阈值
                    if avg_acc > best_acc:
                        best_acc = avg_acc
                        best_thr = thr
                print(f"最佳平均准确率为：{best_acc}, 对应的阈值为：{best_thr}")
                self.test_avg_acc = best_acc
            else:
                y_pred = pred_proba.argmax(axis=1)
                avg_acc = accuracy_score(self.y_test, y_pred)

                print("平均accuracy为：{}, ".format(avg_acc))
                self.test_avg_acc = avg_acc


    def get_average_accuracy(self):
        return self.test_avg_acc

class SolverCDAN(object):
    def __init__(self,  mark, extra, s_train_select, t_path, t_train_select, t_valid_select,  model_save_path, config, clfModel):
        self.config = config
        self.mark = mark # 交叉验证具体哪一折
        self.mode = config.mode  # 判定是哪一个病症

        self.source = config.source  # 源域数据
        self.target = config.target  # 目标域数据

        # 超参数
        self.alpha = config.alpha
        self.beta = config.beta

        self.model_save_path = model_save_path # 保存模型的位置

        self.class_num = config.class_num  # 类别数目（二分类 or 多分类）
        self.gpu = torch.cuda.is_available()  # GPU是否可以使用 默认使用

        # 模型训练相关参数
        self.epoch_num = config.epoch_num
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer  # 优化器
        self.scheduler = config.scheduler  # 迭代器
        self.lr = config.learning_rate
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay  # 权重衰减率
        self.check_epoch = config.check_epoch
        self.early_stop_step = config.early_stop_step
        self.interval = config.interval  # 在训练过程中每隔interval就输出一次train acc之类的结果

        self.slice_size = config.slice_size
        self.modality = config.modality

        self.curr_epoch = 0
        self.feature_dim = config.feature_dim

        # 在迭代中赋值，不需要传参的参数
        self.clf = clfModel # 集成的机器学习分类器
        self.fe_Net = None  # 特征提取器框架
        self.lb_Cls = None  # 分类器框架
        self.da_Net = None  # 域对抗网络
        self.best_acc = 0
        self.break_flag = False


        # print(os.path.basename(__file__))

        # 载入源数据、源标签、目标数据、目标标签
        print("dataset loading......")

        # 根据五折交叉验证 整理s和t相关的数据和标签， s_train_list 和 s_train_label_list; s_val_list 和 s_val_label_list
        s_train_list = s_train_select
        # 这里需要对aug增加特判逻辑
        s_train_label_list = []
        for s in s_train_list:
            if "aug" in s:
                parts = s.split('/')
                # 将'npy_aug'替换为'label'，并移除倒数第二个部分
                new_parts = parts[:-2] + [parts[-1]]
                new_parts[3] = 'label'
                new_path = '/'.join(new_parts)
                s_train_label_list.append(new_path)
            else:
                s_train_label_list.append(s.replace('npy', 'label'))

        # 这里要根据 t_train_select 构建 训练和测试数据对
        print("break point")
        t_train_select_set = set(t_train_select)
        t_train_list = []
        for file_path in os.listdir(t_path):
            if file_path.split('_')[1] in t_train_select_set:
                t_train_list.append(os.path.join(t_path, file_path))

        t_train_label_list = [t.replace('npy', 'label') for t in t_train_list]

        t_valid_select_set = set(t_valid_select)
        t_valid_list = []
        for file_path in os.listdir(t_path):
            if file_path.split('_')[1] in t_valid_select_set and file_path.split('_')[2] == "0":
                t_valid_list.append(os.path.join(t_path, file_path))

        t_valid_label_list = [t.replace('npy', 'label') for t in t_valid_list]

        # 这里判断是否要增加aug部分的数据
        if config.aug == 1:
            t_train_list_aug = []
            t_path_aug = t_path.replace('npy', 'npy_aug')
            for file_path_augs in os.listdir(t_path_aug):
                for file_path in os.listdir(os.path.join(t_path_aug, file_path_augs)):
                    if file_path.split('_')[1] in t_train_select_set:
                        aug = os.path.join(file_path_augs, file_path)
                        t_train_list_aug.append(os.path.join(t_path_aug, aug))
            t_train_label_list_aug = []
            for t in t_train_list_aug:
                parts = t.split('/')
                # 将'npy_aug'替换为'label'，并移除倒数第二个部分
                new_parts = parts[:-2] + [parts[-1]]
                new_parts[3] = 'label'
                new_path = '/'.join(new_parts)
                t_train_label_list_aug.append(new_path)

            # 测试的时候valid 不做增强
            t_train_list.extend(t_train_list_aug)
            t_train_label_list.extend(t_train_label_list_aug)

        # 有oversampling的版本
        # oversampling 策略？
        self.xs_train, self.ys_train, self.xt_train, self.yt_train, self.xt_valid, self.yt_valid  = KflodDataloader(
            self.slice_size,
            self.modality, self.class_num,
            self.mode,
            source_dir=s_train_list,
            label_s_dir=s_train_label_list,
            target_dir=t_train_list,
            label_t_dir=t_train_label_list,
            target_v_dir=t_valid_list,
            label_tv_dir=t_valid_label_list,
        )

        # 增加逻辑 将valid部分拆分为 test 和 valid
        # self.xt_valid, self.yt_valid
        # 这里有问题，可能对某个human重复采样，测算其最后结果，尽可能评估不同病人的 -> 顺序采样即可
        # 这里应该直接前半部分是 valid 后半部分是test
        valid_size = len(self.xt_valid)

        indices = np.arange(valid_size)
        # np.random.shuffle(indices)
        split = int(valid_size * 0.5)  # 50%用于拆分
        self.xt_valid1 = np.array([self.xt_valid[i] for i in indices[:split]])
        self.yt_valid1 = np.array([self.yt_valid[i] for i in indices[:split]])
        self.xt_valid2 = np.array([self.xt_valid[i] for i in indices[split:]])
        self.yt_valid2 = np.array([self.yt_valid[i] for i in indices[split:]])

        x, y, path = clf_data_loader(extra, t_path, t_valid_select)
        splitline = split
        self.x_test = np.array([x[int(i)] for i in indices[:splitline]])
        self.y_test = np.array([y[int(i)] for i in indices[:splitline]])
        x_valid = [x[int(i)] for i in indices[splitline:]]
        y_valid = [y[int(i)] for i in indices[splitline:]]

        print("clf valid 测试：")
        for clf_name, clf_item in self.clf.items():
            y_pred_proba = clf_item.predict_proba(x_valid)
            # 选择概率最高的类别作为预测类别
            y_pred = y_pred_proba.argmax(axis=1)

            # 打印每个分类器的名称（如果分类器没有名称属性，这里可能需要修改）
            print(f"\n分类器：{clf_name}")
            print("clf valid Accuracy:", accuracy_score(y_valid, y_pred))

            # 打印 y_pred 和 y_valid
            print("y_pred:", y_pred)
            print("y_valid:", y_valid)
            print("y_prob:", y_pred_proba)

        # 载入Dataloader
        self.train_dataset = generate_dataset(self.xs_train, self.ys_train, self.xt_train, self.yt_train, self.batch_size, self.gpu)
        self.test_dataset = generate_dataset(self.xt_valid1, self.yt_valid1, self.xt_valid1, self.yt_valid1, self.batch_size, self.gpu)
        self.valid_dataset  =  generate_dataset(self.xt_valid2, self.yt_valid2, self.xt_valid2, self.yt_valid2, self.batch_size, self.gpu)

        len_source = math.ceil(self.ys_train.shape[0] / self.batch_size)
        len_target = math.ceil(self.yt_train.shape[0] / self.batch_size)
        if len_source > len_target:
            self.num_iter = len_source
        else:
            self.num_iter = len_target


    def train_and_valid(self):
        train_acc = []
        valid_acc = []
        train_loss = []
        valid_loss = []

        print("=" * 20 + "mark:{}".format(self.mark) + "=" * 20)

        # 构建网络
        # 特征提取器
        self.fe_Net = eca_resnet20(in_channel=self.modality, modality=self.modality, out_channel=self.feature_dim, device = 'gpu')
        # 域对抗分类器
        self.cdan_Net = CDAN_AdversarialNetwork()
        self.random_layer = RandomLayer([128, self.class_num], 128)

        # 类别分类器
        self.lb_Cls = Label_Classifier(inplane=self.feature_dim, class_num=self.class_num)
        if  self.gpu:
            self.fe_Net.cuda()
            self.lb_Cls.cuda()
            self.cdan_Net.cuda()
            self.random_layer.cuda()
        self.set_optimizer(which_opt=self.optimizer)
        self.set_scheduler(which_sch=self.scheduler)

        # First Train & First Valid
        print('✿' * 10 + '[First Train & First Valid]' + '✿' * 10)
        self.stop_step = 0
        self.break_flag = False
        for epoch in range(self.epoch_num):
            self.curr_epoch = epoch

            start_time = time.time()

            temp_acc1, temp_loss1 = self.train_process(epoch=epoch, dataset=self.train_dataset)

            end_time = time.time()  # 获取结束时间
            print(f"epoch 中训练的时间: {end_time - start_time} 秒")

            # temp_accT, temp_lossT = self.train_processT(epoch=epoch, dataset=self.valid_dataset)
            start_time = time.time()
            # 这里可以尝试更换为根据test的表现 保存最佳的模型
            if self.config.test_save == 1:
                temp_acc2, temp_loss2 = self.test_process(epoch=epoch, dataset=self.test_dataset)
            else:
                temp_acc2, temp_loss2 = self.valid_process(epoch=epoch, dataset=self.valid_dataset)

            end_time = time.time()  # 获取结束时间
            print(f"epoch 中valid的时间: {end_time - start_time} 秒")

            # 如果当前验证集达到最好效果，则保存模型
            # is_greater = all(temp_acc2 > x for x in train_acc)
            # if is_greater:
            #     self.save_model(temp_acc2)
            train_acc.append(temp_acc1)
            valid_acc.append(temp_acc2)
            train_loss.append(temp_loss1)
            valid_loss.append(temp_loss2)
            if self.break_flag:
                break

        # 保存模型
        # self.save_model()

        ### 绘制训练过程中loss和acc变化趋势图 ###
        plt.figure()
        plt.plot(train_acc, 'b', label='Train_acc')
        plt.plot(valid_acc, 'r', label='Valid_acc')
        plt.ylabel('acc')
        plt.xlabel('iter_num')
        plt.legend()
        image_name = str(self.mark) + "acc.jpg"
        plt.savefig(os.path.join(imgPath, image_name))

        plt.figure()
        plt.plot(train_loss, 'b', label='Train_loss')
        plt.plot(valid_loss, 'r', label='Valid_loss')
        plt.ylabel('loss')
        plt.xlabel('iter_num')
        plt.legend()
        image_name = str(self.mark) + "loss.jpg"
        plt.savefig(os.path.join(imgPath, image_name))
        ### 画图 ###

    # 源域训练
    def train_process(self, epoch, dataset):
        """
        :param epoch:  当前轮数
        :param dataset:   数据（可迭代的形式）
        :return:  None
        """
        Acc = []  # 准确率
        Acc_t = []  # 准确率
        Loss = []  # 损失
        Loss_s = []  # 损失
        Loss_t = []  # 损失
        Sens = []  # 敏感度（在所有真样本中预测正确的）
        Prec = []  # 精确度（在所有预测为真的中预测正确的）
        F1 = []  # F1值

        self.model_train()
        self.scheduler_step()

        # 每轮打印 cls loss adv loss align loss 观察变化
        total_cls_loss = 0.0
        total_adv_loss = 0.0
        total_align_loss = 0.0
        total_batches = 0
        for step, train_data in enumerate(dataset):
            xs = train_data['S']
            xt = train_data['T']
            ys = train_data['S_label']
            # print("xs.shape = ", xs.shape)

            if self.gpu:
                xs = xs.cuda()
                xt = xt.cuda()
                ys = ys.cuda().long()

            self.reset_grad()  # 重置梯度值为0

            xs_feature = self.fe_Net(xs)  # 得到的特征
            xt_last = self.fe_Net(xt)

            # 引入条件分布对齐
            xs_out = self.lb_Cls(xs_feature)
            xt_out = self.lb_Cls(xt_last)
            feature_out = torch.cat((xs_out, xt_out), 0)
            softmax_output = nn.Softmax(dim=1)(feature_out)
            domain_feature = torch.cat((xs_feature, xt_last), 0)
            xs_shape = xs_out.shape[0]
            xt_shape = xt_out.shape[0]
            entropy = Entropy(softmax_output)

            adv_loss = CDAN([domain_feature, softmax_output], [xs_shape, xt_shape], self.cdan_Net, entropy,
                            calc_coeff(self.num_iter * epoch + step), self.random_layer)

            # print("--------source feature = ", xs_feature)
            # print("--------target feature = ", xt_last)

            # ----------------------对齐损失------------------- #
            # 尝试引入新的 mcc loss
            target_softmax_out_temp = nn.Softmax(dim=1)(xt_out)
            target_entropy_weight = Entropy(target_softmax_out_temp).detach()
            target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
            target_entropy_weight = self.config.batch_size * target_entropy_weight / torch.sum(target_entropy_weight)
            cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(
                target_softmax_out_temp)
            cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
            mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / self.config.batch_size
            align_loss = mcc_loss
            # cmd = CMD()
            # align_loss = cmd(xs_feature, xt_last)

            # ----------------------交叉熵损失------------------- #
            xs_out = self.lb_Cls(xs_feature)  # 分类结果
            cross_loss = nn.CrossEntropyLoss()
            cls_loss = cross_loss(xs_out, ys)

            # -----------------分类 + 对抗 + 对齐---------------------- #
            # 累加损失值和批次计数
            total_cls_loss += cls_loss.item()
            total_adv_loss += adv_loss.item()
            total_align_loss += self.beta * align_loss.item()
            total_batches += 1

            # 依赖beta*损失函数 实现对抗和分类的调控
            # loss = cls_loss +  adv_loss + align_loss
            self.beta = 1.0
            # 先跑固定的 0.01 0.1 1 10 100
            loss = cls_loss +  adv_loss +  self.beta * align_loss

            acc = accuracy(self.config.thr, xs_out, ys.cpu(), self.class_num, 0)
            Acc.append(acc)
            Loss.append(loss)


            loss.backward()
            self.optimizer_step()

        Acc_last, Loss_last = synthesize(Acc, Loss)
        Loss_last = Loss_last.cpu()
        Loss_last = Loss_last.detach().numpy()
        # print("Loss_last = ", Loss_last)
        # print("Loss_last.type = ", type(Loss_last))
        print('After Epoch ', epoch,  ' :')
        # 打印cls loss 和 adv loss
        avg_cls_loss = total_cls_loss / total_batches
        avg_adv_loss = total_adv_loss / total_batches
        avg_align_loss = total_align_loss / total_batches
        print(
            f'Epoch {epoch}, Average Cls Loss: {avg_cls_loss:.4f}, Average Adv Loss: {avg_adv_loss:.4f}， self.beta:{self.beta}, Average Align Loss: {avg_align_loss:.4f}')
        # 基于损失反馈的动态调整self.beta self.alpha
        # 如何给定调整策略？
        # 设置阈值和调整因子
        # 这块得精心调整
        threshold_increase = 10  # 当align_loss是cls_loss的10倍时开始减少beta
        threshold_decrease = 0.5
        adjust_factor = 0.1  # 减少beta的因子

        # 动态调整beta
        if avg_align_loss / avg_cls_loss > threshold_increase:
            self.beta *= adjust_factor
        if avg_align_loss / avg_cls_loss < threshold_decrease:
            self.beta /= adjust_factor

        # 保证beta在合理范围内 【0.01 0.1】
        self.beta = max(0.1, min(self.beta, 1000))

        print('     [Train S] Acc: {a:.3f}, Loss: {l}'.format(a=Acc_last, l=Loss_last))

        return Acc_last, Loss_last

    # 目标域训练：20%数据 由第五折valid来
    def train_processT(self, epoch, dataset):
        """
        :param epoch:  当前轮数
        :param dataset:   数据（可迭代的形式）
        :return:  None
        """
        Acc = []  # 准确率
        Loss = []  # 损失

        self.model_train()
        self.scheduler_step()

        for step, train_data in enumerate(dataset):
            xt = train_data['T']
            yt = train_data['T_label']

            if self.gpu:
                xt = xt.cuda()
                yt = yt.cuda().long()

            self.reset_grad()  # 重置梯度值为0

            xt_last = self.fe_Net(xt)
            xt_out = self.lb_Cls(xt_last)
            # ----------------------交叉熵损失------------------- #
            cross_loss = nn.CrossEntropyLoss()
            cls_loss = cross_loss(xt_out, yt)
            # ------------------------------------------------- #

            # -----------------只有分类损失---------------------- #
            loss = cls_loss
            acc = accuracy(self.config.thr, xt_out, yt, self.class_num, 0)
            Acc.append(acc)
            Loss.append(loss)

            loss.backward(torch.ones_like(loss))  # 梯度回传
            self.optimizer_step()

        Acc_last, Loss_last = synthesize(Acc, Loss)
        Loss_last = Loss_last.cpu()
        Loss_last = Loss_last.detach().numpy()
        print('     [Train T] Acc: {a:.3f}, Loss: {l}'.format(a=Acc_last, l=Loss_last))

        return Acc_last, Loss_last

    # 验证
    def valid_process(self, epoch, dataset):
        # 这里传入的目标域的valid部分
        Acc = []
        Loss = []
        Sens = []
        Prec = []
        F1 = []

        self.model_eval()

        for step, valid_data in enumerate(dataset):
            xs_v = valid_data['S']
            ys_v = valid_data['S_label']
            if self.gpu:
                xs_v = xs_v.cuda()
                ys_v = ys_v.cuda().long()

            with torch.no_grad():
                xs_v_last = self.fe_Net(xs_v)
                xs_v_out = self.lb_Cls(xs_v_last)
                cross_loss = nn.CrossEntropyLoss()
                cls_loss = cross_loss(xs_v_out, ys_v)
                loss = cls_loss
                acc = accuracy(self.config.thr, xs_v_out, ys_v, self.class_num, 0)
                # acc, Sens, Prec, F1 = accuracy(xs_v_out, ys_v, self.class_num, 0)
                Acc.append(acc)
                Loss.append(loss)
        Acc_v, Loss_v = synthesize(Acc, Loss)

        print('     [Valid] val_Acc: {val_a:.3f}, val_Loss: {val_l:.3f}'.format(val_a=Acc_v, val_l=Loss_v))

        Acc_v_change = round(Acc_v.item(), 3)
        if Acc_v_change > self.best_acc:
            self.stop_step = 0
            self.best_acc = Acc_v_change
            print('    *[Valid Change] Epoch: {e}, Best Acc: {best_acc}'.format(e=epoch, best_acc=self.best_acc))
            self.save_model(Acc_v_change)

        else:
            self.stop_step += 1
        if self.stop_step >= self.early_stop_step:
            print('-' * 40)
            print('The early stopping is triggered at epoch {e}, acc is {a}, loss is {l}'
                  .format(e=epoch, a=Acc_v, l=Loss_v))
            print('-' * 40)
            self.break_flag = True

        Loss_v = Loss_v.cpu()
        Loss_v = Loss_v.detach().numpy()
        return Acc_v, Loss_v

    def test_process(self, epoch, dataset):
        # 这里传入的目标域的test部分 可以同步观测cadn在test集表现
        Acc = []
        Loss = []
        Sens = []
        Prec = []
        F1 = []

        self.model_eval()

        for step, valid_data in enumerate(dataset):
            xs_v = valid_data['S']
            ys_v = valid_data['S_label']
            if self.gpu:
                xs_v = xs_v.cuda()
                ys_v = ys_v.cuda().long()

            with torch.no_grad():
                xs_v_last = self.fe_Net(xs_v)
                xs_v_out = self.lb_Cls(xs_v_last)
                cross_loss = nn.CrossEntropyLoss()
                cls_loss = cross_loss(xs_v_out, ys_v)
                loss = cls_loss
                acc = accuracy(self.config.thr, xs_v_out, ys_v, self.class_num, 0)
                # acc, Sens, Prec, F1 = accuracy(xs_v_out, ys_v, self.class_num, 0)
                Acc.append(acc)
                Loss.append(loss)
        Acc_v, Loss_v = synthesize(Acc, Loss)

        print('     [Test] test_Acc: {val_a:.3f}, test_Loss: {val_l:.3f}'.format(val_a=Acc_v, val_l=Loss_v))

        Acc_v_change = round(Acc_v.item(), 3)
        if Acc_v_change > self.best_acc:
            self.stop_step = 0
            self.best_acc = Acc_v_change
            print('    *[Test Change] Epoch: {e}, Best Acc: {best_acc}'.format(e=epoch, best_acc=self.best_acc))
            self.save_model(Acc_v_change)

        else:
            self.stop_step += 1
        if self.stop_step >= self.early_stop_step:
            print('-' * 40)
            print('The early stopping is triggered at epoch {e}, acc is {a}, loss is {l}'
                  .format(e=epoch, a=Acc_v, l=Loss_v))
            print('-' * 40)
            self.break_flag = True

        Loss_v = Loss_v.cpu()
        Loss_v = Loss_v.detach().numpy()
        return Acc_v, Loss_v

    def test_target(self):
        dataset = self.test_dataset

        Acc = []
        Loss = []
        Sens = []
        Prec = []
        F1 = []

        self.model_eval()

        # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿
        data_out_all = torch.Tensor().cuda()
        label_out_all = torch.LongTensor().cuda()
        # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿
        for step, data in enumerate(dataset):
            # 引入 self clf
            bs = self.config.batch_size
            resDict = {}
            # for clf_name, clf_item in self.clf.items():

            if step*bs+bs < len(self.x_test):
                for clf_name, clf_item in self.clf.items():
                    resDict[clf_name] = clf_item.predict_proba(self.x_test[step*bs: step*bs+bs, :])
            else:
                if self.x_test[step * bs:].shape[0] > 0:
                    for clf_name, clf_item in self.clf.items():
                        resDict[clf_name] = clf_item.predict_proba(self.x_test[step * bs:, :])
                else:
                    continue


            xt = data['T']
            yt = data['T_label']
            if self.gpu:
                xt = xt.cuda()
                yt = yt.cuda().long()

            with torch.no_grad():
                xt_fea = self.fe_Net(xt)
                xt_out = self.lb_Cls(xt_fea)

                # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿
                data_out_all = torch.cat((data_out_all, xt_out), 0)
                label_out_all = torch.cat((label_out_all, yt), 0)
                # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿

                cross_loss = nn.CrossEntropyLoss()
                cls_loss = cross_loss(xt_out, yt)
                loss = cls_loss
                # acc, Sens, Prec, F1 = accuracy(xt_out, yt, self.class_num, 0)

                # acc = accuracy(xt_out, yt, self.class_num, 0)
                acc = accuracyClf(self.config.thr, resDict, xt_out, yt, self.class_num, 0)
                Acc.append(acc)
                Loss.append(loss)

        # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿
        # dict_exist = os.path.exists("./t_SNE")
        # if not dict_exist:
        #     os.makedirs("./t_SNE")
        # data_out_write = data_out_all.cpu().detach().numpy()
        # np.savetxt("./t_SNE/xt_data.txt", data_out_write, fmt="%f")
        # label_out_write = label_out_all.cpu().detach().numpy()
        # np.savetxt("./t_SNE/xt_label.txt", label_out_write, fmt="%f")
        # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿

        Acc_last, Loss_last = synthesize(Acc, Loss)
        print('[Test Target] Acc: {a:.3f}, Loss: {l:.3f}'
              .format(a=Acc_last, l=Loss_last))

        # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿
        # with open('./log/log_so_20201116.txt', 'a') as f:
        #     context = '[Test Target] Acc: {a:.3f}, Loss: {l:.3f}, Sens: {s:.3f}, Spec: {p:.3f}, F1: {f:.3f}'\
        #         .format(a=Acc_last, l=Loss_last, s=Sens_last, p=Prec_last, f=F1_last) + '\n'
        #     f.writelines(context)
        # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿

    def test_source(self):
        dataset = self.test_dataset

        Acc = []
        Loss = []
        Sens = []
        Prec = []
        F1 = []

        self.model_eval()

        # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿
        data_out_all = torch.Tensor().cuda()
        label_out_all = torch.LongTensor().cuda()
        # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿
        print("✿"*20, " source test begin! ", "✿"*20)
        for step, data in enumerate(dataset):
            xs = data['S']
            ys = data['S_label']
            if self.gpu:
                xs = xs.cuda()
                ys = ys.cuda().long()
            with torch.no_grad():
                xs_fea = self.fe_Net(xs)
                xs_out = self.lb_Cls(xs_fea)

                # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿
                data_out_all = torch.cat((data_out_all, xs_out), 0)  # 将两个tensor拼接在一起，0表示按行（第0维）拼接
                label_out_all = torch.cat((label_out_all, ys), 0)
                # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿

                cross_loss = nn.CrossEntropyLoss()
                cls_loss = cross_loss(xs_out, ys)
                loss = cls_loss
                # acc, Sens, Prec, F1 = accuracy(xs_out, ys, self.class_num, 0)
                acc = accuracy1(xs_out, ys, self.class_num, 0)
                Acc.append(acc)
                Loss.append(loss)

        # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿
        # data_out_write = data_out_all.cpu().detach().numpy()  # detach(): 阻断梯度传播
        # np.savetxt("./t_SNE/xs_data.txt", data_out_write, fmt="%f")  # 将得到的源域分类结果写入 xs_data.txt
        # label_out_write = label_out_all.cpu().detach().numpy()
        # np.savetxt("./t_SNE/xs_label.txt", label_out_write, fmt="%f")  # 将读入的源域真实标签写入 xs_label.txt
        # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿

        Acc_last, Loss_last = synthesize(Acc, Loss)
        print('[Test Source] Acc: {a:.3f}, Loss: {l:.3f}'
              .format(a=Acc_last, l=Loss_last))

        # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿
        # with open('./log/log_so_20201116.txt', 'a') as f:
        #     context = '[Test Source] Acc: {a:.3f}, Loss: {l:.3f}, Sens: {s:.3f}, Spec: {p:.3f}, F1: {f:.3f}'\
        #         .format(a=Acc_last, l=Loss_last, s=Sens_last, p=Prec_last, f=F1_last) + '\n'
        #     f.writelines(context)
        # ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿ PRINT ✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿✿

    def set_optimizer(self, which_opt):
        """
        :param which_opt: 选择的优化器的种类（momentum，adam）
        """
        if which_opt == 'momentum':
            self.opt_fe_Net = optim.SGD(self.fe_Net.parameters(),
                                        lr=self.lr,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)

            self.opt_da_Net = optim.SGD(self.da_Net.parameters(),
                                        lr=self.lr,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)

            self.opt_lb_Cls = optim.SGD(self.lb_Cls.parameters(),
                                        lr=self.lr,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)
        elif which_opt == 'adam':
            self.opt_fe_Net = optim.Adam(self.fe_Net.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)

            self.opt_cdan_Net = optim.Adam(self.cdan_Net.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)

            self.opt_lb_Cls = optim.Adam(self.lb_Cls.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)

    def set_scheduler(self, which_sch):
        if which_sch == 'step':
            self.sch_fe_Net = optim.lr_scheduler.StepLR(self.opt_fe_Net, 10, gamma=0.1, last_epoch=-1)
            # 第一个参数就是所使用的优化器对象，第二个参数就是每多少轮循环后更新一次学习率(lr)，第三个参数就是每次更新lr的gamma倍（lr = lr *gamma）
            self.sch_lb_Cls = optim.lr_scheduler.StepLR(self.opt_lb_Cls, 10, gamma=0.1, last_epoch=-1)
        elif which_sch == 'multi_step':
            self.sch_fe_Net = optim.lr_scheduler.MultiStepLR(self.opt_fe_Net, milestones=[5, 10])
            self.sch_lb_Cls = optim.lr_scheduler.MultiStepLR(self.opt_lb_Cls, milestones=[5, 10])
            self.sch_cdan_Net = optim.lr_scheduler.MultiStepLR(self.opt_cdan_Net, milestones=[5, 10])

    def reset_grad(self):
        self.opt_fe_Net.zero_grad()
        self.opt_cdan_Net.zero_grad()
        self.opt_lb_Cls.zero_grad()

    def optimizer_step(self):
        """
        执行优化
        """
        self.opt_fe_Net.step()
        self.opt_cdan_Net.step()
        self.opt_lb_Cls.step()

    def scheduler_step(self):
        self.sch_fe_Net.step()
        self.sch_cdan_Net.step()
        self.sch_lb_Cls.step()

    def model_train(self):
        # 启用 BatchNormalization 和 Dropout
        self.fe_Net.train()
        self.cdan_Net.train()
        self.lb_Cls.train()

    def model_eval(self):
        """
        不启用 BatchNormalization 和 Dropout，保证 BN和 dropout不发生变化，pytorch框架会自动把 BN和 Dropout固定住，
        不会取平均，而是用训练好的值，不然的话，一旦 stest的 batch_size过小，很容易就会被 BN层影响结果。
        """
        self.fe_Net.eval()
        self.cdan_Net.eval()
        self.lb_Cls.eval()

    def save_model(self, acc):
        # 文件名构成：日期-mark-epoch-准确率
        # 获取当前日期时间
        current_datetime = datetime.now()
        # 格式化输出
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        acc = round(acc, 4)

        file_str = formatted_datetime + "--" + str(self.mark) + "--" + str(self.curr_epoch) + "--" + str(acc)
        print(file_str)
        file_name = os.path.join(self.model_save_path, file_str)
        if not os.path.exists(file_name):
            os.makedirs(file_name)

        torch.save(self.fe_Net.state_dict(), os.path.join(file_name, "fe_Net.pkl"))
        torch.save(self.cdan_Net.state_dict(),  os.path.join(file_name, "cdan_Net.pkl"))
        torch.save(self.lb_Cls.state_dict(), os.path.join(file_name, "lb_Cls.pkl"))

    def load_model(self):
        print("Loading Model……")
        self.fe_Net = eca_resnet20(in_channel=self.modality, modality=self.modality, out_channel=self.feature_dim, device = 'gpu')  # 特征提取器
        self.cdan_Net = CDAN_AdversarialNetwork()
        self.lb_Cls = Label_Classifier(inplane=self.feature_dim, class_num=self.class_num)
        if self.gpu:
            self.fe_Net.cuda()
            self.cdan_Net.cuda()
            self.lb_Cls.cuda()
        best_acc = 0
        for file in os.listdir(self.model_save_path):
            result_list = file.split("--")
            mm = result_list[1]
            acc = result_list[3]
            # 按照具体哪一折 mark 进行区分的
            if int(mm) == self.mark and float(acc) >= best_acc:
                best_result = file
                best_acc = float(acc)
        read_path = os.path.join(self.model_save_path, best_result)
        fea_path = os.path.join(read_path, "fe_Net.pkl")
        cls_path = os.path.join(read_path, "lb_Cls.pkl")
        cdan_path = os.path.join(read_path, "cdan_Net.pkl")
        self.fe_Net.load_state_dict(torch.load(fea_path))
        self.cdan_Net.load_state_dict(torch.load(cdan_path))
        self.lb_Cls.load_state_dict(torch.load(cls_path))
        print("Model Load Finish ! ✿ヽ(°▽°)ノ✿")