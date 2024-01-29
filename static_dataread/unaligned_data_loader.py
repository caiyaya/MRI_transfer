import os
import torch.utils.data
import torchnet as tnt
from builtins import object
import torchvision.transforms as transforms
from static_dataread.dataset import Dataset
import torch.utils.data as data

class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = None, None
        B, B_paths = None, None
        try:
            A, A_paths = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths = next(self.data_loader_A_iter)

        try:
            B, B_paths = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S': A, 'S_label': A_paths,
                    'T': B, 'T_label': B_paths}


class UnalignedDataLoader():  # 未对齐的数据加载器
    def initialize(self, source, target, batch_size_xs, batch_size_xt, gpu):
        # if gpu:
        #     xs = torch.from_numpy(source["imgs"]).cuda()
        #     ys = torch.from_numpy(source["label"]).cuda()
        #     xt = torch.from_numpy(target['imgs']).cuda()
        #     yt = torch.from_numpy(target['label']).cuda()
        # else:
        #     xs = torch.from_numpy(source["imgs"])
        #     ys = torch.from_numpy(source["label"])
        #     xt = torch.from_numpy(target['imgs'])
        #     yt = torch.from_numpy(target['label'])

        xs = torch.from_numpy(source["imgs"])
        ys = torch.from_numpy(source["label"])
        xt = torch.from_numpy(target['imgs'])
        yt = torch.from_numpy(target['label'])

        dataset_source = data.TensorDataset(xs, ys)
        dataset_target = data.TensorDataset(xt, yt)

        data_loader_s = data.DataLoader(dataset=dataset_source,
                                        batch_size=batch_size_xs,
                                        shuffle=True)

        data_loader_t = data.DataLoader(dataset=dataset_target,
                                        batch_size=batch_size_xt,
                                        shuffle=True)

        self.dataset_s = dataset_source
        self.dataset_t = dataset_target
        self.paired_data = PairedData(data_loader_s, data_loader_t, float("inf"))

    def initialize_sourceonly(self, source, batch_size_xs, gpu):
        if gpu:
            xs = torch.from_numpy(source["imgs"]).cuda()
            ys = torch.from_numpy(source["label"]).cuda()
        else:
            xs = torch.from_numpy(source["imgs"])
            ys = torch.from_numpy(source["label"])

        dataset_source = data.TensorDataset(xs, ys)

        data_loader_s = data.DataLoader(dataset=dataset_source,
                                        batch_size=batch_size_xs,
                                        shuffle=True, transforms=transformers)
        self.dataset_s = data_loader_s

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def load_data_sourceonly(self):
        return self.dataset_s

    def __len__(self):
        return min(max(len(self.dataset_s), len(self.dataset_t)), float("inf"))
