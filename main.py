# 五折交叉，测试
from solver.solver_Kflod_tripleloss import *
# 全部用来训练
# from solver.HX_Kflod_train import *
# 测试版
# from solver.tmp_solver import *
import numpy as np
from configs.config import *
from print2log import *
from sklearn.model_selection import KFold, StratifiedKFold
import argparse
from sklearn import svm
from sklearn.metrics import accuracy_score
from static_dataread.dataset_read import clf_data_loader


# hyper-parameter
parser = argparse.ArgumentParser("Number Transfer")
parser.add_argument('-c', '--config', default='./configs/number_config.json')
args = parser.parse_args()
config = process_config(args.config)
# 设置随机种子
np.random.seed(1024)


def main():
    model_save_path = './model_log'  # 模型存储路径
    source = r"./data_new/mice/npy"
    target = r"./data_new/human/npy"
    extra = r"./data_new/human_extra" # 89个 需要按照batch 扩增

    # 增加 s_ids 用来做train 和 test 的分割
    s_path = []
    s_ids = set()
    for file_path in os.listdir(source):
        s_ids.add(file_path.split('_')[1])
        s_path.append(os.path.join(source, file_path))

    t_path = []
    t_ids = set()
    for file_path in os.listdir(target):
        t_ids.add(file_path.split('_')[1])
        t_path.append(os.path.join(target, file_path))

    # 如果需要使用aug数据
    if config.aug == 1:
        source_aug = r"./data_new/mice/npy_aug"
        for file_augs in os.listdir(source_aug):
            for file_path in os.listdir(os.path.join(source_aug, file_augs)):
                file_path = os.path.join(file_augs, file_path)
                s_path.append(os.path.join(source_aug, file_path))


    # -------------------五折划分目标域训练集、验证集和测试集（源域全部用于训练）--------------------
    mark = 1
    kf = KFold(n_splits=5, shuffle=True) # 五折相当于0.2，后续可以调整
    # for (t_train_index, t_valid_index) in kf.split(t_path):
    t_ids = list(t_ids)
    for (t_train_index, t_valid_index) in kf.split(list(t_ids)):
        print("-----------------------------------------")
        print("The "+str(mark)+"th ZHE experiment")
        print("Select target validset and testset（0.2）:", t_valid_index)
        # 这里根据ids 来对t_path 进行分割
        t_train_select, t_valid_select = np.array(t_ids)[t_train_index], np.array(t_ids)[t_valid_index]

        # 提取train中id 对应的 参数信息
        # svm训练 参数信息

        x_train, y_train = clf_data_loader(extra, target, t_train_select)
        clf = svm.SVC(kernel='poly', degree=3, probability=True)  # 使用线性核
        clf.fit(x_train, y_train)

        solver = Solver2(mark=mark, extra=extra, s_train_select=s_path, t_path=target, t_train_select=t_train_select, t_valid_select=t_valid_select, model_save_path=model_save_path,
                        config=config, clfModel=clf)

        solver.train_and_valid()
        solver.load_model()

        print('✿' * 70)
        # solver.test_source()
        solver.test_target()
        print('✿' * 46)
        print("\n")

        mark += 1


if __name__ == '__main__':
    make_print_to_file(config=config, path='./text_log')
    main()  # 如果只训练&测试一次，记得退回一行缩进
