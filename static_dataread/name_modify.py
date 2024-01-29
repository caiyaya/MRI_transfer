import os
import re
# path = r"D:\\study\\code\\nafld_transfer\\data\\youyi_mice_data"
# path = r"D:\\study\\code\\nafld_transfer\\data\\youyi_mice_label"
path = r"D:\\study\\code\\nafld_transfer\\data\\human_data"
# path = r"D:\\study\\code\\nafld_transfer\\data\\human_label"
name_list = os.listdir(path)
print(name_list)
for i in name_list:
    old = path + os.sep + i
    print(old)
    new = path + os.sep + 'human_' + re.findall("\d+", i)[0]
    os.rename(old, new)
