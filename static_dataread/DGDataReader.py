from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
from static_dataread.data_utils import *
import random
import cv2


class FourierDGDataset(Dataset):
    def __init__(self, data, labels, transformer=None, alpha=1.0):
        
        self.data = data
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()  # 归一化
        self.alpha = alpha
        # assert len(self.data) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self):
        img_pa = self.data
        label_pa = self.labels
        img = []
        label = []
        for i in range(len(img_pa)):
            img_ss = self.transformer(img_pa[i])  # 取一个病例的三维切片
            label_ss = label_pa[i]
            label_same_list = []
            for j in range(len(label_pa)):
                if label_pa[j] == label_ss:
                    label_same_list.append(j)
            img_oo = self.transformer(img_s[random.sample(label_same_list, 1)])  # 取一个同病例的三维切片
            label_oo = label_ss
            for slice in range(len(img_ss[slice])):
                img_o = img_oo[slice]
                img_s = img_ss[slice]
                # img_s2o = alpha * img_s + (1 - alpha) * img_o
                img_s2o, img_o2s = colorful_spectrum_mix(img_o, img_s, alpha=self.alpha)  # 他们有着相同的label
                img_o, img_s = self.post_transform(img_o), self.post_transform(img_s)
                img_s2o, img_o2s = self.post_transform(img_s2o), self.post_transform(img_o2s)
                img.append(img_s2o)
            img = np.array(img)
            img_final = np.concatenate((img_pa, img), axis=0)
            label = np.concatenate((label_pa, label_ss), axis=0)
        assert len(img_final) == len(self.label)
        return img_final, label


# def get_dataset(path, train=False, image_size=224, crop=False, jitter=0, config=None):
#     names, labels = dataset_info(path)
#     if config:
#         image_size = config["image_size"]
#         crop = config["use_crop"]
#         jitter = config["jitter"]
#     img_transform = get_img_transform(train, image_size, crop, jitter)
#     return DGDataset(names, labels, img_transform)

def get_post_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return img_transform


def get_fourier_dataset(data, labels, image_size=128, crop=False, jitter=0, alpha=1.0, config=None):
    assert len(data) == len(labels)
    img_transformer = get_pre_transform(image_size, crop, jitter)

    img_pa = data
    label_pa = labels
    img = []
    label = []
    for i in range(len(img_pa)):
        img_ss = img_pa[i]  # 取一个病例的三维切片
        label_ss = label_pa[i]
        label_same_list = []
        for j in range(len(label_pa)):
            if label_pa[j] == label_ss:
                label_same_list.append(j)
        img_oo = img_transformer(img_s[random.sample(label_same_list, 1)])  # 取一个同病例的三维切片
        for slice in range(len(img_ss[slice])):
            img_o = img_oo[slice]
            img_s = img_ss[slice]
            # img_s2o = alpha * img_s + (1 - alpha) * img_o
            img_s2o, img_o2s = colorful_spectrum_mix(img_o, img_s, alpha=alpha)  # 他们有着相同的label
            img_o, img_s = get_post_transform(img_o), get_post_transform(img_s)
            img_s2o, img_o2s = get_post_transform(img_s2o), get_post_transform(img_o2s)
            img.append(img_s2o)
        img = np.array(img)
        img_final = np.concatenate((img_pa, img), axis=0)
        label = np.concatenate((label_pa, label_ss), axis=0)
    assert len(img_final) == len(label)
    return img_final, label
