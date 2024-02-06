# 五折交叉，测试
from solver.solver_Kflod_tripleloss import *
from solver.solver_Kflod_DCAN import *
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
import time

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# hyper-parameter
parser = argparse.ArgumentParser("Number Transfer")
parser.add_argument('-c', '--config', default='./configs/number_config.json')
args = parser.parse_args()
config = process_config(args.config)
# 设置随机种子
np.random.seed(45)


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
        # # # 这里增加判断，看需要五折中的哪几折
        # if(file_path.split('_')[2] == "2"):
        t_path.append(os.path.join(target, file_path))

    # 如果需要使用aug数据
    if config.aug == 1:
        source_aug = r"./data_new/mice/npy_aug"
        for file_augs in os.listdir(source_aug):
            for file_path in os.listdir(os.path.join(source_aug, file_augs)):
                file_path = os.path.join(file_augs, file_path)
                s_path.append(os.path.join(source_aug, file_path))

    # 创建分类器
    clf = {
        'svm': svm.SVC(kernel='poly', degree=3, probability=True),
        'lr': LogisticRegression(),
        'dt': DecisionTreeClassifier(),
        'rf': RandomForestClassifier(),
        'gb': GradientBoostingClassifier(),
        'knn': KNeighborsClassifier(n_neighbors=4),
        'nb': GaussianNB(),
    }

    # -------------------五折划分目标域训练集、验证集和测试集（源域全部用于训练）--------------------
    mark = 1
    kf = KFold(n_splits=5, shuffle=True) # 五折相当于0.2，后续可以调整
    # for (t_train_index, t_valid_index) in kf.split(t_path):
    t_ids = list(t_ids)
    for (t_train_index, t_valid_index) in kf.split(list(t_ids)):
        # 埋点，测算代码耗时部分
        start_time = time.time()

        print("-----------------------------------------")
        print("The "+str(mark)+"th ZHE experiment")
        print("Select target validset and testset（0.2）:", t_valid_index)
        # 这里根据ids 来对t_path 进行分割
        t_train_select, t_valid_select = np.array(t_ids)[t_train_index], np.array(t_ids)[t_valid_index]

        # 提取train中id 对应的 参数信息
        # svm训练 参数信息
        # 引入更多的分类器
        x_train, y_train = clf_data_loader(extra, target, t_train_select)
        print("y_train:", y_train)

        for clf_name, clf_item in clf.items():
            clf_item.fit(x_train, y_train)

        # 埋点1
        end_time = time.time()  # 获取结束时间
        print(f"svm对于 train 部分的训练时间: {end_time - start_time} 秒")
        start_time = time.time()

        # solver = Solver2(mark=mark, extra=extra, s_train_select=s_path, t_path=target, t_train_select=t_train_select, t_valid_select=t_valid_select, model_save_path=model_save_path,
        #                 config=config, clfModel=clf)

        # 使用cdAn的
        solver = SolverCDAN(mark=mark, extra=extra, s_train_select=s_path, t_path=target, t_train_select=t_train_select,
                         t_valid_select=t_valid_select, model_save_path=model_save_path,
                         config=config, clfModel=clf)

        # 埋点10
        end_time = time.time()  # 获取结束时间
        print(f"SolverCDAN准备时间: {end_time - start_time} 秒")


        solver.train_and_valid()
        solver.load_model()

        print('✿' * 70)
        # solver.test_source()
        solver.test_target()
        print('✿' * 46)
        print("\n")

        mark += 1


if __name__ == '__main__':
    # 清除./model_log/目录下的所有文件
    model_log_dir = './model_log/'
    if os.path.exists(model_log_dir):
        for filename in os.listdir(model_log_dir):
            file_path = os.path.join(model_log_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}\n')
    # 删除后建立新的即可
    os.makedirs(model_log_dir, exist_ok=True)

    make_print_to_file(config=config, path='./text_log')
    main()  # 如果只训练&测试一次，记得退回一行缩进
