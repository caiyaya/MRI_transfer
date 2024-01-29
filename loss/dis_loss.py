from functools import partial
import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import Function

from utils import *


class MMD(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss


class CMD(nn.Module):
    def __init__(self, moments=5):
        super(CMD, self).__init__()
        self.moments = moments

    def matchnorm(self, source, target):
        eps = 1e-6
        return ((source-target)**2 + eps).sum().sqrt()

    def scm(self, s_source, s_target, k):
        ss1 = (s_source**int(k)).mean(0)
        ss2 = (s_target**int(k)).mean(0)
        return self.matchnorm(ss1, ss2)

    def forward(self, source, target):
        m_source = source.mean(0)
        m_target = target.mean(0)
        s_source = source - m_source
        s_target = target - m_target
        dm = self.matchnorm(m_source, m_target)
        scms = dm
        for i in range(self.moments - 1):
            scms = scms + self.scm(s_source, s_target, i + 2)
        loss = scms
        return loss


def cls_dis_loss(s, t, ys, pseudo_yt, gpu):
    # xs_c0, xs_c1, xs_c2 = data_group(s, ys, gpu)
    # xt_c0, xt_c1, xt_c2 = data_group(t, pseudo_yt, gpu)
    #
    # mmd = MMD()
    #
    # if xt_c0.shape[0] != 0:
    #     dis_loss_0 = mmd.forward(xs_c0, xt_c0)
    # else:
    #     dis_loss_0 = 0
    #
    # if xt_c1.shape[0] != 0:
    #     dis_loss_1 = mmd.forward(xs_c1, xt_c1)
    # else:
    #     dis_loss_1 = 0
    #
    # if xt_c2.shape[0] != 0:
    #     dis_loss_2 = mmd.forward(xs_c2, xt_c2)
    # else:
    #     dis_loss_2 = 0
    #
    # dis_loss = (dis_loss_0 + dis_loss_1 + dis_loss_2)/3

    # 💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩 shit code 2 💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩
    xs_c0, xs_c1, xs_c2, xs_c3, xs_c4, xs_c5, xs_c6, xs_c7, xs_c8, xs_c9 = data_group(s, ys, gpu)
    xt_c0, xt_c1, xt_c2, xt_c3, xt_c4, xt_c5, xt_c6, xt_c7, xt_c8, xt_c9 = data_group(t, pseudo_yt, gpu)

    mmd = MMD()

    if xt_c0.shape[0] != 0:
        dis_loss_0 = mmd.forward(xs_c0, xt_c0)
    else:
        dis_loss_0 = 0

    if xt_c1.shape[0] != 0:
        dis_loss_1 = mmd.forward(xs_c1, xt_c1)
    else:
        dis_loss_1 = 0

    if xt_c2.shape[0] != 0:
        dis_loss_2 = mmd.forward(xs_c2, xt_c2)
    else:
        dis_loss_2 = 0

    if xt_c3.shape[0] != 0:
        dis_loss_3 = mmd.forward(xs_c3, xt_c3)
    else:
        dis_loss_3 = 0

    if xt_c4.shape[0] != 0:
        dis_loss_4 = mmd.forward(xs_c4, xt_c4)
    else:
        dis_loss_4 = 0

    if xt_c5.shape[0] != 0:
        dis_loss_5 = mmd.forward(xs_c5, xt_c5)
    else:
        dis_loss_5 = 0

    if xt_c6.shape[0] != 0:
        dis_loss_6 = mmd.forward(xs_c6, xt_c6)
    else:
        dis_loss_6 = 0

    if xt_c7.shape[0] != 0:
        dis_loss_7 = mmd.forward(xs_c7, xt_c7)
    else:
        dis_loss_7 = 0

    if xt_c8.shape[0] != 0:
        dis_loss_8 = mmd.forward(xs_c8, xt_c8)
    else:
        dis_loss_8 = 0

    if xt_c9.shape[0] != 0:
        dis_loss_9 = mmd.forward(xs_c9, xt_c9)
    else:
        dis_loss_9 = 0

    dis_loss = (dis_loss_0 + dis_loss_1 + dis_loss_2 + dis_loss_3
                + dis_loss_4 + dis_loss_5 + dis_loss_6 + dis_loss_7
                + dis_loss_8 + dis_loss_9)/10

    return dis_loss
