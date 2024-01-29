import numpy as np
import os

from torch.utils.data import Dataset


class GenerateDataset(Dataset):
    def __init__(self, root_path, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        self.root_path = root_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 从数据列表中获取图像路径、文字描述和类别标签
        image_path, description, label = self.data_list[idx]

        # 读取npy数据
        path = os.path.join(self.root_path, image_path)
        data = np.load(path)
        data = np.transpose(data, (1, 2, 0))  # 进行ToTensor变化的时候会
        description = np.array(description)

        # 对图像应用变换（裁切、旋转等）
        if self.transform:
            data = self.transform(data)

        label = int(label)

        # return image, description, label
        return data, description, label, self.data_list


class CustomDataset(Dataset):
    def __init__(self, root_path, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        self.root_path = root_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 从数据列表中获取图像路径、文字描述和类别标签
        image_path, label = self.data_list[idx]

        # 读取npy数据
        path = os.path.join(self.root_path, image_path)
        data = np.load(path)
        data = np.transpose(data, (1, 2, 0))  # 进行ToTensor变化的时候会调整维度，那我提前先打乱

        # 对图像应用变换（裁切、旋转等）
        if self.transform:
            data = self.transform(data)

        label = int(label)

        # return image, description, label
        return data, label, self.data_list

