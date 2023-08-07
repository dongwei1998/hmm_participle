# coding=utf-8
# =============================================
# @Time      : 2022-07-20 17:08
# @Author    : DongWei1998
# @FileName  : parameter.py
# @Software  : PyCharm
# =============================================
import os
from easydict import EasyDict
from dotenv import load_dotenv,find_dotenv
import logging.config
import shutil



# 创建路径
def check_directory(path, create=True):
    flag = os.path.exists(path)
    if not flag:
        if create:
            os.makedirs(path)
            flag = True
    return flag


def parser_opt(model):
    load_dotenv(find_dotenv())  # 将.env文件中的变量加载到环境变量中
    args = EasyDict()
    logging.config.fileConfig(os.environ.get("logging_ini"))
    args.logger = logging.getLogger('model_log')
    # 清除模型以及可视化文件
    if model == 'train':
        args.train_data_file = os.path.join(os.environ.get('data_files'),'pku_training.utf8')
        args.output_dir = os.environ.get('output_dir')



    elif model =='env':
        pass
    elif model == 'server':
        pass
    else:
        raise print('请给定model参数，可选【traian env test】')
    return args


if __name__ == '__main__':
    args = parser_opt('train')