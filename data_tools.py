import os

import SimpleITK as sitk
# import numpy as np
# import nibabel as nib  # nii格式一般都会用到这个包
import pandas as pd
import openpyxl


# ----------------------统一mask的文件名----------------------- #
# mask_path = r"/mnt/E/hx/data/youyi_human/youyi_human_mask"
# for every_pa in os.listdir(mask_path):
#     tmp_mask_path_1 = os.path.join(mask_path, every_pa)
#     for every_mo in os.listdir(tmp_mask_path_1):
#         old = os.path.join(tmp_mask_path_1, every_mo)
#         if "T1WI" in every_mo:
#             new = os.path.join(tmp_mask_path_1, "1T1WI.nii.gz")
#             os.rename(old, new)
#         if "T2WI" in every_mo:
#             new = os.path.join(tmp_mask_path_1, "2T2WI.nii.gz")
#             os.rename(old, new)
#         if "IQ" in every_mo:
#             new = os.path.join(tmp_mask_path_1, "3IDEAL IQ.nii.gz")
#             os.rename(old, new)
#         if "MRE" in every_mo:
#             new = os.path.join(tmp_mask_path_1, "4MRE.nii.gz")
#             os.rename(old, new)
#         if "DWI" in every_mo:
#             new = os.path.join(tmp_mask_path_1, "5DWI.nii.gz")
#             os.rename(old, new)
#         if "6" in every_mo:
#             new = os.path.join(tmp_mask_path_1, "6T1mapping.nii.gz")
#             os.rename(old, new)
#         if "dce" in every_mo or "7" in every_mo:
#             new = os.path.join(tmp_mask_path_1, "7DCE.nii.gz")
#             os.rename(old, new)
#         print(old, " finish")
# ------------------------------------------------------------------ #

# ----------------------统一slice的文件名----------------------- #
mask_path = r"/mnt/E/hx/data/youyi_human/youyi_human_original"
for every_pa in os.listdir(mask_path):
    tmp_mask_path_1 = os.path.join(mask_path, every_pa)
    for every_mo in os.listdir(tmp_mask_path_1):
        old = os.path.join(tmp_mask_path_1, every_mo)
        if "T1WI" in every_mo:
            new = os.path.join(tmp_mask_path_1, "1T1WI")
            os.rename(old, new)
        if "T2WI" in every_mo:
            new = os.path.join(tmp_mask_path_1, "2T2WI")
            os.rename(old, new)
        if "IQ" in every_mo:
            new = os.path.join(tmp_mask_path_1, "3IDEAL IQ")
            os.rename(old, new)
        if "MRE" in every_mo:
            new = os.path.join(tmp_mask_path_1, "4MRE")
            os.rename(old, new)
        if "DWI" in every_mo:
            new = os.path.join(tmp_mask_path_1, "5DWI")
            os.rename(old, new)
        if "6" in every_mo:
            new = os.path.join(tmp_mask_path_1, "6T1mapping")
            os.rename(old, new)
        if "dce" in every_mo or "7" in every_mo:
            new = os.path.join(tmp_mask_path_1, "7DCE")
            os.rename(old, new)
        print(old, " finish")
# ------------------------------------------------------------------ #

# # -------------------重命名病人文件夹(序号连着的情况)---------------------- #
# path = r"D:\data\youyi_human\94-100_TMP"
# i = 94
# for every_pa in os.listdir(path):
#     old = os.path.join(path, every_pa)
#     ii = str(i)
#     new = os.path.join(path, "human_"+ii)
#     os.rename(old, new)
#     i = i + 1

# path = r"D:\data\youyi\youyi_human\youyi_human_original"
# for every_pa in os.listdir(path):
#     old = os.path.join(path, every_pa)
#     new = os.path.join(path, every_pa[0: -5]+"_aug1")
#     os.rename(old, new)

# path = r"D:\data\youyi\youyi_human\youyi_human_original"
# for every_pa in os.listdir(path):
#     old = os.path.join(path, every_pa)
#     new = os.path.join(path, "human_"+every_pa)
#     print(new)
#     os.rename(old, new)
# # ------------------------------------------------------------------ #

# # ------------------------重命名病人文件夹(序号不连着的情况)------------------------ #
# path = r"D:\data\youyi\youyi_human\update_DWI"
# for every_pa in os.listdir(path):
#     print(every_pa)
#     tmp = every_pa[5:]
#     every_pa1 = 'human_' + tmp
#     # print(tmp)
#     old = os.path.join(path, every_pa)
#     new = os.path.join(path, every_pa1)
#     os.rename(old, new)
# ------------------------------------------------------------------------------ #

# -------------------------------将病人编号写入excel------------------------------- #
# read_path = r"D:\data\youyi_mice\mice_dicom\mice_dcm"
# excel_path = r"C:\Users\Han\Desktop\mice_list.xlsx"
# list = []
# for every_pa in os.listdir(read_path):
#     print(every_pa)
#     tmp = every_pa[5:7]
#     print(tmp)
#     list.append(tmp)
# print(list)
# wb = openpyxl.Workbook()
# ws = wb.active
# for r in range(len(list)):
#     ws.cell(r + 1, 1).value = list[r]
# wb.save(excel_path)
# print("成功写入文件: " + excel_path + " !")
# ------------------------------------------------------------------------------ #

# -----------------------删除某一模态--------------------------------- #
# path = r"/home/lab321/D/hx/EMCA_net/data/mice_mm_fib"
# for every_pa in os.listdir(path):
#     tmp_path_1 = os.path.join(path, every_pa)
#     for every_mo in os.listdir(tmp_path_1):
#         if "map" in every_mo or "DWI" in every_mo or "IVIM" in every_mo:
#             path_1 = os.path.join(tmp_path_1, every_mo)
#             os.remove(path_1)

# path = r"C:\Users\Han\Desktop\human_64"
# for every_pa in os.listdir(path):
#     tmp_path_1 = os.path.join(path, every_pa)
#     for every_mo in os.listdir(tmp_path_1):
#         if "ADC" in every_mo:
#             path_1 = os.path.join(tmp_path_1, every_mo)
#             os.remove(path_1)

# path = r"D:\aaa"
# for every_pa in os.listdir(path):
#     tmp_path_1 = os.path.join(path, every_pa)
#     for every_mo in os.listdir(tmp_path_1):
#         if "DWI" in every_mo:
#             path_1 = os.path.join(tmp_path_1, every_mo)
#             os.remove(path_1)
# # ------------------------------------------------------------------ #



# -----------------------处理DWI（只取一部分DWI的切片）--------------------------------- #
# path = r"E:\data\youyi_human\human_nii_DWI"
# for every_pa in os.listdir(path):
#     tmp_path_1 = os.path.join(path, every_pa)  # human//human_1
#     for every_mo in os.listdir(tmp_path_1):
#         if "DKI" in every_mo:
#             tmp_path_2 = os.path.join(tmp_path_1, every_mo)
#             img_livermask = sitk.ReadImage(tmp_path_2)  # 切片序列
#             img_liverarr = sitk.GetArrayFromImage(img_livermask)  # 三维数组
#             slice_num = img_liverarr.shape[0]
#             # print(tmp_path_2, " slice num = ", slice_num)
#         elif "DWI" in every_mo:
#             tmp_path_2 = os.path.join(tmp_path_1, every_mo)
#             img = nib.load(tmp_path_2)
#             imgvol = np.array(img.dataobj)
#             imgvol = imgvol[:, :, 0:slice_num]  # 只保存0-slicenum部分的切片
#             newimg = nib.Nifti1Image(imgvol, img.affine)
#             os.remove(tmp_path_2)
#             newimg.to_filename(tmp_path_2)

# -----------------------------检查原图和mask切片数是否一致-------------------------------- #
# slice_path = r"D:\data\youyi_mice\priority_processing\mice_nii"
# mask_path = r"D:\data\youyi_mice\priority_processing\mice_mask"
# for every_pa in os.listdir(slice_path):
#     tmp_slice_1 = os.path.join(slice_path, every_pa)
#     tmp_mask_1 = os.path.join(mask_path, every_pa)
#     for every_nii in os.listdir(tmp_slice_1):
#         tmp_slice_2 = os.path.join(tmp_slice_1, every_nii)
#         if '1T1WI' in every_nii or 'T2' in every_nii or 'MRE' in every_nii or 'mapping' in every_nii:
#             tmp_mask_2 = os.path.join(tmp_mask_1, every_nii + ".gz")  # equal MaskPath\\mice(11)\\1T1WI.nii.gz
#         elif 'pdff' in every_nii or 'R2' in every_nii:
#             tmp_mask_2 = os.path.join(tmp_mask_1, "3IDEAL IQ.nii.gz")
#         elif 'ADC' in every_nii or ("DKI" in every_nii) or ("IVIM" in every_nii):
#             tmp_mask_2 = os.path.join(tmp_mask_1, "5DWI.nii.gz")
#         elif "7" in every_nii:
#             tmp_mask_2 = os.path.join(tmp_mask_1, "7DCE.nii.gz")
#         else:
#             continue
#         img = sitk.ReadImage(tmp_slice_2)  # 切片序列
#         img_arr = sitk.GetArrayFromImage(img)  # 切片三维数组
#         slice_num = img_arr.shape[0]
#
#         img_livermask = sitk.ReadImage(tmp_mask_2)  # 切片序列
#         img_liverarr = sitk.GetArrayFromImage(img_livermask)  # 三维数组
#         mask_num = img_liverarr.shape[0]
#
#         if not slice_num == mask_num:
#             print("病例 ", every_pa, " 的模态 ", every_nii, " 切片数目不匹配")
#             print("slice number is ", slice_num)
#             print("mask number is ", mask_num)
# # ---------------------------------------------------------------------------- #

# -----------------新建label文件夹并将excel中的标签信息自动写入txt--------------------- #
# excel存储格式见readme文件夹下的“label示例excel格式.xlsx"
# save_path = r"D:\data\youyi_mice\priority_processing\mice_label"
# excel_path = r"D:\data\youyi_mice\priority_processing\mice_label.xlsx"
# pa_num = 40  # 待处理的病例总数
# if not os.path.exists(save_path):
#     print("save path is not exist!")
# else:
#     excel_data = pd.read_excel(excel_path)
#     data = excel_data.values
#     # print(data)
#     # print(data[:, 0])  # 是个行向量
#     pa_no = data[:, 0]
#     pa_fib = data[:, 1]
#     pa_fat = data[:, 2]
#     pa_hep = data[:, 3]
#     for i in range(0, pa_num):
#         num = str(pa_no[i])
#         fib = str(pa_fib[i])
#         fat = str(pa_fat[i])
#         hep = str(pa_hep[i])
#         file_path = os.path.join(save_path, "mice_"+num)
#         os.makedirs(file_path)
#         txt_path = os.path.join(file_path, "label.txt")
#         file = open(txt_path, 'w')
#         content = fib+','+fat+','+hep
#         print("content = ", content)
#         file.write(content)
#         file.close()
# # ---------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------ #


