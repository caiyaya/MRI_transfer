import json
import subprocess
import sys
# 定义参数的值
batch_sizes = [12, 24, 36]
learning_rates = [1e-1, 1e-2, 1e-3]
early_stop_steps = [5, 10, 25]

# number.json文件路径
json_file_path = 'configs/number_config.json'
log_file_path = 'log_new.log'

# 打开日志文件
with open(log_file_path, 'w') as log_file:
    # 遍历每个参数的值
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for early_stop_step in early_stop_steps:
                # 读取JSON文件
                with open(json_file_path, 'r') as file:
                    data = json.load(file)

                # 修改参数的值
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
                subprocess.run(command, stdout=log_file, stderr=log_file)

                # 打印分隔符以区分每次执行的输出
                log_file.write('-' * 50 + '\n')
                log_file.flush()  # 确保写入文件
