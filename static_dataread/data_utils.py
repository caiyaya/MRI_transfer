from torchvision import transforms
import random
import torch
import numpy as np
from math import sqrt


# 读取文件列表
# 文件列表格式：路径 标签
def dataset_info(filepath):
    with open(filepath, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.strip().split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def get_img_transform(train=False, image_size=224, crop=False, jitter=0):
    # image_size：调整完的图像尺寸；crop：是否进行图像裁剪；jitter：对图像属性的修改幅度
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        if crop:
            img_transform = [transforms.RandomResizedCrop(image_size, scale=[0.8, 1.0])]
        else:
            img_transform = [transforms.Resize((image_size, image_size))]
        if jitter > 0:
            # transforms.ColorJitter:改变图像的属性：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue)
            # eg: brightness_change = transforms.ColorJitter(brightness=0.5):它的含义是将图像的亮度随机变化为原图亮度的50%（1−0.5）∼150%（1+0.5）
            img_transform.append(transforms.ColorJitter(brightness=jitter,
                                                        contrast=jitter,
                                                        saturation=jitter,
                                                        hue=min(0.5, jitter)))
        # transform.ToTensor()能够把灰度范围从0-255变换到0-1之间
        # transform.Normalize()则把0-1变换到(-1,1)
        img_transform += [transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize(mean, std)]
        # 一般用Compose把多个步骤整合到一起
        img_transform = transforms.Compose(img_transform)
    else:
        img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return img_transform

# 传入参数是非tensor时的情况，操作分别为随即裁切，色彩畸变，随机水平翻转
def get_pre_transform(image_size=224, crop=False, jitter=0):
    if crop:
        img_transform = [transforms.ToTensor(), transforms.RandomResizedCrop(image_size, scale=[0.8, 1.0])]
    else:
        img_transform = [transforms.ToTensor(), transforms.Resize((image_size, image_size))]
    if jitter > 0:
        img_transform.append(transforms.ColorJitter(brightness=jitter,
                                                    contrast=jitter,
                                                    saturation=jitter,
                                                    hue=min(0.5, jitter)))
    img_transform += [transforms.RandomHorizontalFlip()]
    img_transform = transforms.Compose(img_transform)
    return img_transform


#
def get_post_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return img_transform


def get_spectrum(img):
    # 计算二维的傅里叶变换
    img_fft = np.fft.fft2(img)
    # 计算振幅img_abs与复数对应的角度（相位）img_pha
    img_abs = np.abs(img_fft)
    img_pha = np.angle(img_fft)
    return img_abs, img_pha


def get_centralized_spectrum(img):
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)  # 将FFT输出中的直流分量移动到频谱中央
    img_abs = np.abs(img_fft)
    img_pha = np.angle(img_fft)
    return img_abs, img_pha


def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """
    频谱混合
    Input image size: ndarray of [H, W, C]
    alpha：两个图片融合比例的上限
    """
    lam = np.random.uniform(0, alpha)  # 从一个均匀分布[low,high)中随机采样，采样数量为size（size缺省则为输出一个值）

    assert img1.shape == img2.shape  # 发生异常就在这里停止并报错，防止崩溃
    h, w, c = img1.shape
    # 裁出来中间的一块
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12