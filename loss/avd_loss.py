import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


def group_adv_loss(xs, xt):
    domain_pred = torch.cat([xs, xt])
    domain_y_ind = torch.cat([torch.ones(xs.shape[0], dtype=torch.long), torch.zeros(xt.shape[0], dtype=torch.long)])
    domain_y_ind = domain_y_ind.cuda()
    cross_loss = nn.CrossEntropyLoss()
    g_adv_loss = cross_loss(domain_pred, domain_y_ind)

    # domain_y = F.one_hot(domain_y_ind, num_classes=2).float()
    # g_adv_loss = F.binary_cross_entropy_with_logits(domain_pred, domain_y)

    return g_adv_loss


def domain_adv_loss(s, t, ys, pseudo_yt, gpu):
    # xs_c0, xs_c1, xs_c2 = data_group(s, ys, gpu)
    # xt_c0, xt_c1, xt_c2 = data_group(t, pseudo_yt, gpu)
    #
    # if (len(xs_c0)==0) & (len(xt_c0)==0):
    #     g_adv_loss_0 = 0
    # else:
    #     g_adv_loss_0 = group_adv_loss(xs_c0, xt_c0)
    #
    # if (len(xs_c1)==0) & (len(xt_c1)==0):
    #     g_adv_loss_1 = 0
    # else:
    #     g_adv_loss_1 = group_adv_loss(xs_c1, xt_c1)
    #
    # if (len(xs_c2)==0) & (len(xt_c2)==0):
    #     g_adv_loss_2 = 0
    # else:
    #     g_adv_loss_2 = group_adv_loss(xs_c2, xt_c2)
    #
    # g_adv_loss = (g_adv_loss_0 + g_adv_loss_1 + g_adv_loss_2)/3

    # 💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩 shit code 3 💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩💩
    xs_c0, xs_c1, xs_c2, xs_c3, xs_c4, xs_c5, xs_c6, xs_c7, xs_c8, xs_c9 = data_group(s, ys, gpu)
    xt_c0, xt_c1, xt_c2, xt_c3, xt_c4, xt_c5, xt_c6, xt_c7, xt_c8, xt_c9 = data_group(t, pseudo_yt, gpu)

    if (len(xs_c0)==0) & (len(xt_c0)==0):
        g_adv_loss_0 = 0
    else:
        g_adv_loss_0 = group_adv_loss(xs_c0, xt_c0)

    if (len(xs_c1)==0) & (len(xt_c1)==0):
        g_adv_loss_1 = 0
    else:
        g_adv_loss_1 = group_adv_loss(xs_c1, xt_c1)

    if (len(xs_c2)==0) & (len(xt_c2)==0):
        g_adv_loss_2 = 0
    else:
        g_adv_loss_2 = group_adv_loss(xs_c2, xt_c2)

    if (len(xs_c3)==0) & (len(xt_c3)==0):
        g_adv_loss_3 = 0
    else:
        g_adv_loss_3 = group_adv_loss(xs_c3, xt_c3)

    if (len(xs_c4)==0) & (len(xt_c4)==0):
        g_adv_loss_4 = 0
    else:
        g_adv_loss_4 = group_adv_loss(xs_c4, xt_c4)

    if (len(xs_c5)==0) & (len(xt_c5)==0):
        g_adv_loss_5 = 0
    else:
        g_adv_loss_5 = group_adv_loss(xs_c5, xt_c5)

    if (len(xs_c6)==0) & (len(xt_c6)==0):
        g_adv_loss_6 = 0
    else:
        g_adv_loss_6 = group_adv_loss(xs_c6, xt_c6)

    if (len(xs_c7)==0) & (len(xt_c7)==0):
        g_adv_loss_7 = 0
    else:
        g_adv_loss_7 = group_adv_loss(xs_c7, xt_c7)

    if (len(xs_c8)==0) & (len(xt_c8)==0):
        g_adv_loss_8 = 0
    else:
        g_adv_loss_8 = group_adv_loss(xs_c8, xt_c8)

    if (len(xs_c9)==0) & (len(xt_c9)==0):
        g_adv_loss_9 = 0
    else:
        g_adv_loss_9 = group_adv_loss(xs_c9, xt_c9)

    g_adv_loss = (g_adv_loss_0 + g_adv_loss_1 + g_adv_loss_2 + g_adv_loss_3
                  + g_adv_loss_4 + g_adv_loss_5 + g_adv_loss_6 + g_adv_loss_7
                  + g_adv_loss_8 + g_adv_loss_9)/10

    return g_adv_loss