import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

"""
将单张npy文件转换为png图片
输入：folderpath：npy文件存储路径（E:\\temporary\\test\mice_npy\mice (88)\\7DCE_Vp.nii.npy）
     savepath：png结果存储路径（E:\\temporary\\test）精确到存储文件夹位置
     默认png文件名称为folderpath中的npy文件名
"""


# 将npy转为单通道png
def npy_png(folderpath, savepath):
    arr = np.load(folderpath)
    im = Image.fromarray(arr)
    im = im.convert('L')  # 转为灰度图(保持单通道)
    print(type(im))
    (a, b) = os.path.split(folderpath)
    save_path = os.path.join(savepath, b)
    save_path = save_path + ".png"
    print(save_path)
    plt.imsave(save_path, im)
    print("存好啦")

    '''
    path=dest_dir + "_" + '{:06d}'.format(count) + ".png"
    image = Image.open(path)
    image = image.convert("RGB")　# 转换为RGB
    os.remove(path)
    image.save(path)　# RGB图片替换此灰度图
    '''


# 将npy转为三通道png
# 待显示numpy已经归一化到0-255了
def npy2png_unnorm(folderpath, savepath):
    loadData = np.load(folderpath)
    # list = [loadData.shape[0], loadData.shape[1], loadData.shape[2]]
    if not np.any(loadData):
        print("It is empty numpy!")
    (a, b) = os.path.split(folderpath)
    save_path = os.path.join(savepath, b)
    save_path = save_path + ".png"
    print(save_path)
    plt.imsave(save_path, loadData, cmap='plasma')  # 定义命名规则，保存图片
    # cv2.imwrite(save_path, disp_to_img)
    print("存好啦")


# 将npy转为三通道png
# 待显示numpy是个二值图
def npy2png_norm(folderpath, savepath):
    loadData = np.load(folderpath)
    # list = [loadData.shape[0], loadData.shape[1]]
    for i in range(0, loadData.shape[0]):
        for j in range(0, loadData.shape[1]):
            if loadData[i][j] == 1:
                loadData[i][j] = 255
            elif loadData[i][j] == 0:
                loadData[i][j] = 0
            else:
                print("这不是个二值图啊！")
    if not np.any(loadData):
        print("It is empty numpy!")
    (a, b) = os.path.split(folderpath)
    save_path = os.path.join(savepath, b)
    save_path = save_path + ".png"
    print(save_path)
    cv2.imwrite(save_path, loadData)


def npy2png_5(folderpath, savepath):
    loadData = np.load(folderpath)
    list = [loadData.shape[0], loadData.shape[1], loadData.shape[2]]
    for i in range(0, list[2]):
        # 将result中的第三张切片保存为png格式的图片
        data = loadData[:, :, i]
        (a, b) = os.path.split(folderpath)
        save_path = os.path.join(savepath, b)
        save_path = save_path + str(i) + ".png"
        print(save_path)
        cv2.imwrite(save_path, data)


# png格式的图片转化为npy（内含将npy归一到0-255的过程）
def png2npy(folderpath, savepath):
    img = cv2.imread(folderpath)  # -1表示按照图片原有格式进行读取
    npy = img[:, :, 1]
    print('npy.shape = ', npy.shape)
    # 归一化到0-255




# 对人和大鼠数据进行转换
if __name__ == "__main__":
    folderPath = r"D:\data\youyi\youyi_human\human_png_fourier_1"
    savePath = r"D:\data\youyi\youyi_human\human_npy_fourier_1"
    for every_mice in os.listdir(folderPath):
        tmp_mice_path = os.path.join(folderPath, every_mice)  # E:\\temporary\\test\mice_npy\mice(11)
        tmp_save_path = os.path.join(savePath, every_mice)  # E:\\temporary\\test\png\mice(11)
        if not os.path.exists(tmp_save_path):
            os.makedirs(tmp_save_path)
        for every_MR in os.listdir(tmp_mice_path):
            tmp_path = os.path.join(tmp_mice_path, every_MR)
            png2npy(tmp_path, tmp_save_path)
            # npy2png_unnorm(tmp_path, tmp_save_path)  # 三通道
            # npy2png_norm(tmp_path, tmp_save_path)  # 三通道
            # npy_png(tmp_path, tmp_save_path)  # 单通道
            print("-----", tmp_save_path, " finish-----")
#             if every_MR == "2T2WI.nii.npy":
#                 tmp_path = os.path.join(tmp_mice_path, every_MR)
#                 npy2png_5(tmp_path, tmp_save_path)
#                 print("-----", tmp_path, " finish-----")
#             else:
#                 tmp_path = os.path.join(tmp_mice_path, every_MR)
#                 npy2png(tmp_path, tmp_save_path)
#                 print("-----", tmp_path, " finish-----")

# 对一张图片进行转换
# if __name__ == "__main__":
#     path = r"D:\data\youyi\youyi_human\test\png\human_001\1T1WI.nii.npy.png"
#     savePath = r"D:\data\youyi\youyi_human\test\png\human_001"
#     # npy2png_unnorm(path, savePath)


# if __name__ == "__main__":
#     folderPath = r"D:\Coding\my_code\EMCA_net\data\human_mm_data_inflammation\human_1\1T1WI.nii.npy"
#     savePath = r'D:\Coding\my_code\EMCA_net\data\human_mm_data_inflammation_aug'
#     # for every_mice in os.listdir(folderPath):
#     #     tmp_mice_path = os.path.join(folderPath, every_mice)  # E:\\temporary\\test\mice_npy\mice(11)
#     #     tmp_save_path = os.path.join(savePath, every_mice)  # E:\\temporary\\test\png\mice(11)
#     #     if not os.path.exists(tmp_save_path):
#     #         os.makedirs(tmp_save_path)
#     #     for every_MR in os.listdir(tmp_mice_path):
#     #         if every_MR == "2T2WI.nii.npy":
#     #             tmp_path = os.path.join(tmp_mice_path, every_MR)
#     npy2png(folderPath, savePath)
#     print("-----", folderPath, " finish-----")

