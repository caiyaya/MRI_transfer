import torch
import torch.nn as nn


class CMD_Loss(nn.Module):
    def __init__(self, class_num, n_moments=3):
        super(CMD_Loss, self).__init__()
        self.class_num = class_num
        self.n_moments = n_moments

    def matchnorm(self, x1, x2):
        return torch.sqrt(torch.sum(torch.pow(x1 - x2, 2)))

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

    def cmd(self, x1, x2):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(self.n_moments - 1):
            scms = scms + self.scm(sx1, sx2, i + 2)
        return scms
