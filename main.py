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


# hyper-parameter
parser = argparse.ArgumentParser("Number Transfer")
parser.add_argument('-c', '--config', default='./configs/number_config.json')
args = parser.parse_args()
config = process_config(args.config)
# 设置随机种子
np.random.seed(1024)


def main():
    model_save_path = './model_log'  # 模型存储路径
    source = r"./data/mice/npy_128"
    target = r"./data/human/npy_128"

    s_path = []
    for file_path in os.listdir(source):
        s_path.append(os.path.join(source, file_path))

    t_path = []
    for file_path in os.listdir(target):
        t_path.append(os.path.join(target, file_path))

    # -------------------五折划分训练集和验证集：源领域和目标领域--------------------
    mark = 1
    kf = KFold(n_splits=5, shuffle=True)
    for (t_train_index, t_valid_index) in kf.split(t_path):
        print("-----------------------------------------")
        print("The "+str(mark)+"th ZHE experiment")
        print("Select target valid set:", t_valid_index)
        t_train_select, t_valid_select = np.array(t_path)[t_train_index], np.array(t_path)[t_valid_index]

        solver = Solver2(mark=mark, s_train_select = s_path, t_train_select=t_train_select, t_valid_select=t_valid_select, model_save_path=model_save_path,
                        config=config)

        solver.train_and_valid()
        solver.load_model()  # 注释第26行，调这里

        print('✿' * 70)
        solver.test_source()
        solver.test_target()
        print('✿' * 46)
        print("\n")

        mark += 1


if __name__ == '__main__':
    make_print_to_file(config=config, path='./text_log')
    main()  # 如果只训练&测试一次，记得退回一行缩进
