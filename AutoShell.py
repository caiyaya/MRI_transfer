import json
import subprocess
import sys
import os
import shutil

# 定义参数的值
batch_sizes = [12, 24, 36]
learning_rates = [1e-1, 1e-2, 1e-3]
early_stop_steps = [5, 10, 25]

# number.json文件路径
json_file_path = 'configs/number_config.json'
log_file_path = 'log_new.log'
model_log_dir = './model_log/'  # 日志保存路径

# 打开日志文件
with open(log_file_path, 'w') as log_file:
    # 遍历每个参数的值
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for early_stop_step in early_stop_steps:
                # 读取JSON文件并修改参数的值
                with open(json_file_path, 'r') as file:
                    data = json.load(file)

                data['batch_size'] = batch_size
                data['learning_rate'] = learning_rate
                data['early_stop_step'] = early_stop_step

                # 将修改后的数据写回JSON文件
                with open(json_file_path, 'w') as file:
                    json.dump(data, file, indent=4)

                # 构建命令和日志消息
                command = [sys.executable, 'main.py']
                log_message = f"Running main.py with batch_size={batch_size}, learning_rate={learning_rate}, early_stop_step={early_stop_step}\n"

                # 执行main.py并重定向输出到日志文件
                log_file.write(log_message)
                log_file.flush()  # 确保写入文件

                try:
                    subprocess.run(command, check=True, stdout=log_file, stderr=log_file)
                except subprocess.CalledProcessError as e:
                    # 如果main.py执行失败，记录错误信息
                    log_file.write(f"Error running main.py: {e}\n")

                # 清除./model_log/目录下的所有文件
                if os.path.exists(model_log_dir):
                    for filename in os.listdir(model_log_dir):
                        file_path = os.path.join(model_log_dir, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            log_file.write(f'Failed to delete {file_path}. Reason: {e}\n')

                # 打印分隔符以区分每次执行的输出
                log_file.write('-' * 50 + '\n')
                log_file.flush()  # 确保写入文件
