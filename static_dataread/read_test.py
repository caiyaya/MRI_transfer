import json
import os




def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):  # 拼接fold, image
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):  # 拼接label
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]
