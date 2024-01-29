import sys
import os
import datetime


class Logger(object):
    def __init__(self, filename, path="./"):
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def make_print_to_file(config, path='./'):
    fileName = config.source + config.target + "_" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60, '*'))