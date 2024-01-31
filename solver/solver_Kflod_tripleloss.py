import os
import torch
import torch.optim as optim
from datetime import datetime

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

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

imgPath = "./caicaiImage/img"

class Solver2(object):
    def __init__(self, mark, s_train_select, t_path, t_train_select, t_valid_select,  model_save_path, config):

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
        s_train_label_list = [s.replace('npy', 'label') for s in s_train_list]

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
            if file_path.split('_')[1] in t_valid_select_set:
                t_valid_list.append(os.path.join(t_path, file_path))

        t_valid_label_list = [t.replace('npy', 'label') for t in t_valid_list]


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

        # --------------------------------------------------------------------------------------------------------- #
        # print("loading finished!")

        # print("ys_train.shape = ", self.ys_train.shape[0])
        # print("ys_test.shape = ", self.ys_test.shape[0])
        # print("yt_train.shape = ", self.yt_train.shape[0])
        # print("yt_test.shape = ", self.yt_test.shape[0])

        # 增加逻辑 将valid部分拆分为 test 和 valid
        # self.xt_valid, self.yt_valid
        valid_size = len(self.xt_valid)

        indices = np.arange(valid_size)
        np.random.shuffle(indices)
        split = int(valid_size * 0.5)  # 50%用于拆分
        self.xt_valid1 = np.array([self.xt_valid[i] for i in indices[:split]])
        self.yt_valid1 = np.array([self.yt_valid[i] for i in indices[:split]])
        self.xt_valid2 = np.array([self.xt_valid[i] for i in indices[split:]])
        self.yt_valid2 = np.array([self.yt_valid[i] for i in indices[split:]])

        # 载入Dataloader
        self.train_dataset = generate_dataset(self.xs_train, self.ys_train, self.xt_train, self.yt_train, self.batch_size, self.gpu)
        self.valid_dataset = generate_dataset(self.xt_valid1, self.yt_valid1, self.xt_valid1, self.yt_valid1, self.batch_size, self.gpu)
        self.test_dataset =  generate_dataset(self.xt_valid2, self.yt_valid2, self.xt_valid2, self.yt_valid2, self.batch_size, self.gpu)


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
        self.da_Net = Domain_Adversarial_Net()
        # 类别分类器
        self.lb_Cls = Label_Classifier(inplane=self.feature_dim, class_num=self.class_num)
        if  self.gpu:
            self.fe_Net.cuda()
            self.da_Net.cuda()
            self.lb_Cls.cuda()
        self.set_optimizer(which_opt=self.optimizer)
        self.set_scheduler(which_sch=self.scheduler)

        # First Train & First Valid
        print('✿' * 10 + '[First Train & First Valid]' + '✿' * 10)
        self.stop_step = 0
        self.break_flag = False
        for epoch in range(self.epoch_num):
            self.curr_epoch = epoch
            temp_acc1, temp_loss1 = self.train_process(epoch=epoch, dataset=self.train_dataset)
            # temp_accT, temp_lossT = self.train_processT(epoch=epoch, dataset=self.valid_dataset)
            temp_acc2, temp_loss2 = self.valid_process(epoch=epoch, dataset=self.valid_dataset)
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
            # print("--------source feature = ", xs_feature)
            # print("--------target feature = ", xt_last)

            # ----------------------对齐损失------------------- #
            # cmd = CMD()
            # align_loss = cmd(xs_last, xt_last)

            # ----------------------对抗损失------------------- #
            # todo: 引入loss函数更新 + 加入互信息正则化项（类别敏感的正则化项）
            xs_adv = self.da_Net(xs_feature)
            xt_adv = self.da_Net(xt_last)
            adv_loss = avd_loss.group_adv_loss(xs_adv, xt_adv)

            # ----------------------三元组损失------------------- #
            # margin = 0.2
            # triple_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin)).cuda()
            # # ------------------------------------------------- #
            # ys_t = torch.t(ys)
            # loss_fn = triple_fn(xs_feature, ys_t)
            xs_out = self.lb_Cls(xs_feature)  # 分类结果
            # ----------------------交叉熵损失------------------- #
            cross_loss = nn.CrossEntropyLoss()
            cls_loss = cross_loss(xs_out, ys)
            # ------------------------------------------------- #

            # -----------------只有分类损失---------------------- #
            # loss = cls_loss
            # -----------------分类+对抗---------------------- #
            loss = cls_loss + self.beta * adv_loss
            # print("***********************************adv:", adv_loss)
            # -----------------分类+对齐---------------------- #
            # loss = cls_loss + self.alpha * align_loss
            # print("***********************align:", align_loss)
            # -----------------只有三元组损失---------------------- #
            # loss = loss_fn

            # -----------------分类+三元组损失---------------------- #
            # loss = cls_loss + loss_fn

            # if epoch >= 6:
            #     loss = cls_loss + loss_fn
            # else:
            #     loss = loss_fn

            # print("loss = ", loss)
            # acc, sens, prec, f1 = accuracy(xs_out, ys, self.class_num, 0)
            # ++++++++++++++++++++++++临时+++++++++++++++++++++++++ #
            acc = accuracy(xs_out, ys.cpu(), self.class_num, 0)
            Acc.append(acc)
            Loss.append(loss)

            # Sens.append(sens)
            # Prec.append(prec)
            # F1.append(f1)

            # if step % self.interval == 0:  # 每进行interval个样例训练之后输出一次[Train] Epoch
            #     print('[Train] Epoch: {e}, Batch: {b}, Accuracy: {a:.3f}, Loss:{l:.3f}, '
            #           .format(e=epoch, b=step, a=acc, l=loss))

            loss.backward()
            # loss.backward(torch.ones_like(loss))  # 梯度反传
            self.optimizer_step()

        # Acc_last, Loss_last, Sens_last, Prec_last, F1_last = synthesize(Acc, Loss, Sens, Prec, F1)
        # print('[Train Result] Acc: {a:.3f}, Loss: {l:.3f}, Sens: {s:.3f}, Spec: {p:.3f}, F1: {f:.3f} '
        #       .format(a=Acc_last, l=Loss_last, s=Sens_last, p=Prec_last, f=F1_last))

        Acc_last, Loss_last = synthesize(Acc, Loss)
        Loss_last = Loss_last.cpu()
        Loss_last = Loss_last.detach().numpy()
        # print("Loss_last = ", Loss_last)
        # print("Loss_last.type = ", type(Loss_last))
        print('After Epoch ', epoch,  ' :')
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
            acc = accuracy(xt_out, yt, self.class_num, 0)
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
                acc = accuracy(xs_v_out, ys_v, self.class_num, 0)
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
                acc = accuracy1(xt_out, yt, self.class_num, 0)
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

            self.opt_da_Net = optim.Adam(self.da_Net.parameters(),
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
            self.sch_fe_Net = optim.lr_scheduler.MultiStepLR(self.opt_fe_Net, milestones=[100, 150])
            self.sch_lb_Cls = optim.lr_scheduler.MultiStepLR(self.opt_lb_Cls, milestones=[100, 150])

    def reset_grad(self):
        self.opt_fe_Net.zero_grad()
        self.opt_lb_Cls.zero_grad()

    def optimizer_step(self):
        """
        执行优化
        """
        self.opt_fe_Net.step()
        self.opt_lb_Cls.step()

    def scheduler_step(self):
        self.sch_fe_Net.step()
        self.sch_lb_Cls.step()

    def model_train(self):
        # 启用 BatchNormalization 和 Dropout
        self.fe_Net.train()
        self.da_Net.train()
        self.lb_Cls.train()

    def model_eval(self):
        """
        不启用 BatchNormalization 和 Dropout，保证 BN和 dropout不发生变化，pytorch框架会自动把 BN和 Dropout固定住，
        不会取平均，而是用训练好的值，不然的话，一旦 stest的 batch_size过小，很容易就会被 BN层影响结果。
        """
        self.fe_Net.eval()
        self.da_Net.eval()
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
        # torch.save(self.da_Net.state_dict(), './model_log/da_Net.pkl')
        torch.save(self.lb_Cls.state_dict(), os.path.join(file_name, "lb_Cls.pkl"))

    def load_model(self):
        print("Loading Model……")
        self.fe_Net = eca_resnet20(in_channel=self.modality, modality=self.modality, out_channel=self.feature_dim, device = 'gpu')  # 特征提取器
        # self.da_Net = Domain_Adversarial_Net()
        self.lb_Cls = Label_Classifier(inplane=self.feature_dim, class_num=self.class_num)
        if self.gpu:
            self.fe_Net.cuda()
            # self.da_Net.cuda()
            self.lb_Cls.cuda()
        best_acc = 0
        for file in os.listdir(self.model_save_path):
            result_list = file.split("--")
            mm = result_list[1]
            acc = result_list[3]
            if int(mm) == self.mark and float(acc) >= best_acc:
                best_result = file
                best_acc = float(acc)
        # 这里看下 best_result 是否有问题
        read_path = os.path.join(self.model_save_path, best_result)
        fea_path = os.path.join(read_path, "fe_Net.pkl")
        cls_path = os.path.join(read_path, "lb_Cls.pkl")
        self.fe_Net.load_state_dict(torch.load(fea_path))
        # self.da_Net.load_state_dict(torch.load('./model_log/da_Net.pkl'))
        self.lb_Cls.load_state_dict(torch.load(cls_path))
        print("Model Load Finish ! ✿ヽ(°▽°)ノ✿")