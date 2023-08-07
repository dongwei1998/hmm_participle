# coding=utf-8
# =============================================
# @Time      : 2022-08-02 9:03
# @Author    : DongWei1998
# @FileName  : train.py
# @Software  : PyCharm
# =============================================

from utils import parameter
import numpy as np
import os



def normalize(a):
    tmp = np.log(np.sum(a))
    for i in range(len(a)):
        if a[i] == 0:
            a[i] = float(-2 ** 31)
        else:
            a[i] = np.log(a[i]) - tmp
    return a
def fit(train_file_path, encoding='utf-8'):
    """
    基于给定好的数据进行分词的HMM模型训练
    :param train_file_path:
    :param encoding:
    :return:
    """
    # 1. 读取数据
    with open(train_file_path, mode='r', encoding=encoding) as reader:
        sentence = reader.read()[1:]

    # 2. 初始化相关概率值
    """
    隐状态4个，观测值65536个
    隐状态:
        B: 表示一个单词的开始，0
        M：表示单词的中间，1
        E：表示单词的结尾，2
        S：表示一个字形成一个单词，3
    """
    pi = np.zeros(4)
    A = np.zeros((4, 4))
    B = np.zeros((4, 65536))

    # 3. 模型训练(遍历数据即可)
    # 初始的隐状态为2
    last_state = 2
    # 基于空格对数据做一个划分
    tokens = sentence.split(" ")
    # 迭代处理所有的单词
    for token in tokens:
        # 除去单词前后空格
        token = token.strip()
        # 获取单词的长度
        length = len(token)
        # 过滤异常的单词
        if length < 1:
            continue

        # 处理长度为1的单词, 也就是一个字形成的单词
        if length == 1:
            pi[3] += 1
            A[last_state][3] += 1
            # ord的函数作用是获取字符状态为ACSII码
            B[3][ord(token[0])] += 1
            last_state = 3
        else:
            # 如果长度大于1，那么表示这个词语至少有两个单词，那么初始概率中为0的增加1
            pi[0] += 1

            # 更新状态转移概率矩阵
            A[last_state][0] += 1
            if length == 2:
                # 这个词语只有两个字组成
                A[0][2] += 1
            else:
                # 这个词语至少三个字组成
                A[0][1] += 1
                A[1][2] += 1
                A[1][1] += (length - 3)

            # 更新隐状态到观测值的概率矩阵
            B[0][ord(token[0])] += 1
            B[2][ord(token[-1])] += 1
            for i in range(1, length - 1):
                B[1][ord(token[i])] += 1

            last_state = 2

    # 4. 计算概率值
    pi = normalize(pi)
    for i in range(4):
        A[i] = normalize(A[i])
        B[i] = normalize(B[i])

    return pi, A, B

if __name__ == '__main__':
    args = parameter.parser_opt(model='train')
    pi, A, B = fit(args.train_data_file)
    # 模型参数保存
    np.save(os.path.join(args.output_dir,'pi.npy'),pi)
    np.save(os.path.join(args.output_dir,'A.npy'),A)
    np.save(os.path.join(args.output_dir,'B.npy'),B)