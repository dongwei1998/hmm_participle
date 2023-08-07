# coding=utf-8
# =============================================
# @Time      : 2022-08-02 9:51
# @Author    : DongWei1998
# @FileName  : hmm_learn.py
# @Software  : PyCharm
# =============================================


import numpy as np


def log_sum_exp(arr):
    # a. 将数据转换为numpy数组
    arr = np.array(arr)
    # b. 获取最大值
    max_v = max(arr)
    # c. 计算最大值和最小值差值的指数函数值的和
    tmp = np.sum(np.exp(arr - max_v))
    # d. 求解最终值并返回
    return max_v + np.log(tmp)


def calc_alpha(pi, A, B, Q, alpha=None):
    """
    根据传入的参数计算前向概率alpha
    :param pi:  初始时刻隐状态的概率值向量，是进行对数转换之后的概率值
    :param A:  状态与状态之间的转移概率矩阵A，是进行对数转换之后的概率值
    :param B:  状态和观测值之间的转移概率矩阵B，是进行对数转换之后的概率值
    :param Q:  观测值序列集合，eg: [0,1,0,0,1]
    :param alpha:  前向概率矩阵，是进行对数转换之后的概率值
    :return:  计算好的前向概率矩阵，是进行对数转换之后的概率值
    """
    # 1. 参数初始化
    n = np.shape(A)[0]
    T = np.shape(Q)[0]
    if alpha is None:
        alpha = np.zeros(shape=(T, n))

    # 2. 定义/计算初始时刻t=0的时候前向概率值
    for i in range(n):
        alpha[0][i] = pi[i] + B[i][Q[0]]

    # 3. 计算t=1到t=T-1时刻的前向概率值
    for t in range(1, T):
        for i in range(n):
            # a. 计算累加值
            tmp_prob = np.zeros(shape=n)
            for j in range(n):
                tmp_prob[j] = alpha[t - 1][j] + A[j][i]
            # b. 基于累加值计算当前时刻t对应状态i的前向概率值
            alpha[t][i] = log_sum_exp(tmp_prob) + B[i][Q[t]]

    # 4. 结果返回
    return alpha


def calc_beta(pi, A, B, Q, beta=None):
    """
    根据传入的参数计算后向概率beta
     :param pi:  初始时刻隐状态的概率值向量，是进行对数转换之后的概率值
    :param A:  状态与状态之间的转移概率矩阵A，是进行对数转换之后的概率值
    :param B:  状态和观测值之间的转移概率矩阵B，是进行对数转换之后的概率值
    :param Q:  观测值序列集合，eg: [0,1,0,0,1]
    :param beta: 后向概率矩阵，是进行对数转换之后的概率值
    :return: 返回计算好的后向概率矩阵，是进行对数转换之后的概率值
    """
    # 1. 参数初始化
    n = np.shape(A)[0]
    T = np.shape(Q)[0]
    if beta is None:
        beta = np.zeros(shape=(T, n))

    # 2. 计算t=T-1时刻对应的后向概率值
    for i in range(n):
        beta[T - 1][i] = 0

    # 3. 迭代计算t=T-2到t=0时刻对应的后向概率值
    for t in range(T - 2, -1, -1):
        for i in range(n):
            # a. 计算累加值
            tmp_prob = np.zeros(shape=n)
            for j in range(n):
                tmp_prob[j] = A[i][j] + beta[t + 1][j] + B[j][Q[t + 1]]
            # b. 计算后向概率值
            beta[t][i] = log_sum_exp(tmp_prob)
    # 4. 结果返回
    return beta


def calc_gamma(alpha, beta, gamma=None):
    """
    根据传入的参数计算概率矩阵gamma
    :param alpha:  前向概率矩阵，是进行对数转换之后的概率值
    :param beta:  后向概率矩阵，是进行对数转换之后的概率值
    :param gamma:  gamma概率矩阵，是进行对数转换之后的概率值
    :return:  返回计算后的结果，是进行对数转换之后的概率值
    """
    # 1. 参数初始化
    T, n = np.shape(alpha)
    if gamma is None:
        gamma = np.zeros(shape=(T, n))

    # 2.迭代更新gamma矩阵的值
    # a. 计算p(Q)的概率值
    tmp_prob = log_sum_exp(alpha[-1])
    # b. 计算gamma值
    for t in range(T):
        for i in range(n):
            gamma[t][i] = alpha[t][i] + beta[t][i] - tmp_prob
    # c. 结果返回
    return gamma


def calc_ksi(alpha, beta, A, B, Q, ksi=None):
    """
    根据传入的参数计算概率矩阵gamma
    :param alpha:  前向概率矩阵
    :param beta:  后向概率矩阵
    :param A:  状态与状态之间的转移概率矩阵A
    :param B:  状态和观测值之间的转移概率矩阵B
    :param Q:  观测值序列集合，eg: [0,1,0,0,1]
    :param ksi:  ksi概率矩阵
    :return:  返回计算后的结果
    """
    # 1. 参数初始化
    T, n = np.shape(alpha)
    if ksi is None:
        ksi = np.zeros(shape=(T - 1, n, n))

    # 2.迭代更新kesi矩阵的值
    # a. 计算p(Q)的概率值
    tmp_prob = log_sum_exp(alpha[-1])
    # b. 计算kesi值
    for t in range(T - 1):
        for i in range(n):
            for j in range(n):
                ksi[t][i][j] = alpha[t][i] + A[i][j] + B[j][Q[t + 1]] + beta[t + 1][j] - tmp_prob
    # c. 结果返回
    return ksi


def baum_welch(pi, A, B, Q, max_iter=3):
    # 1. 参数初始化
    T = np.shape(Q)[0]
    n = np.shape(A)[0]
    m = np.shape(B)[1]
    alpha = np.zeros((T, n))
    beta = np.zeros((T, n))
    gamma = np.zeros((T, n))
    ksi = np.zeros((T - 1, n, n))

    # 2. 迭代更新
    for time in range(max_iter):
        # E步：计算相关概率值
        calc_alpha(pi, A, B, Q, alpha)
        calc_beta(pi, A, B, Q, beta)
        calc_gamma(alpha, beta, gamma)
        calc_ksi(alpha, beta, A, B, Q, ksi)

        # M步：模型参数更新，让似然函数最大化
        # b1. 更新pi的值
        for i in range(n):
            pi[i] = gamma[0][i]
        # b2. 更新A值
        for i in range(n):
            for j in range(n):
                # a. 计算分子和分母的值
                a = np.zeros(T - 1)
                b = np.zeros(T - 1)
                for t in range(T - 1):
                    a[t] = gamma[t][i]
                    b[t] = ksi[t][i][j]
                # b. 计算给定的分子、分母计算概率值
                A[i][j] = log_sum_exp(b) - log_sum_exp(a)
        # b3. 更新B
        for i in range(n):
            for j in range(m):
                # a. 计算分子和分母的值
                a = np.zeros(T)
                b = np.zeros(T)
                number = 0
                for t in range(T):
                    a[t] = gamma[t][i]
                    if Q[t] == j:
                        b[number] = ksi[t][i][j]
                        number += 1
                # b. 计算给定的分子、分母计算概率值
                if number == 0:
                    B[i][j] = float(-2 ** 31)
                else:
                    B[i][j] = log_sum_exp(b[:number]) - log_sum_exp(a)
    return pi, A, B


def viterbi(pi, A, B, Q, delta=None):
    """
    根据传入的参数计算delta概率矩阵以及最优可能的状态序列以及出现的最大概率值
    :param pi:
    :param A:
    :param B:
    :param Q:
    :param delta:
    :return:
    """
    # 1. 初始化参数
    n = np.shape(A)[0]
    T = np.shape(Q)[0]
    if delta is None:
        delta = np.zeros(shape=(T, n))
    # pre_index[t][i]表示的是t时刻状态为i的最有可能的上一个时刻的状态值
    pre_index = np.zeros((T, n), dtype=np.int64)

    # 2. 求解t=0时刻对应的delta值
    for i in range(n):
        delta[0][i] = pi[i] + B[i][Q[0]]

    # 3. 求解t=1到t=T-1时刻对应的delta值
    for t in range(1, T):
        for i in range(n):
            # a. 获取最大值
            max_delta = delta[t - 1][0] + A[0][i]
            for j in range(1, n):
                tmp = delta[t - 1][j] + A[j][i]
                if tmp > max_delta:
                    max_delta = tmp
                    pre_index[t][i] = j

            # b. 基于最大概率值计算delta值
            delta[t][i] = max_delta + B[i][Q[t]]

    # 4. 获取最有可能的序列
    decode = np.ones(shape=T, dtype=np.int64) * -1
    # 首先找出最后一个时刻对应的索引下标
    max_delta_index = np.argmax(delta[-1])
    max_prob = delta[-1][max_delta_index]
    decode[-1] = max_delta_index
    # 基于最后一个时刻的最优状态，反退前面时刻的最大概率状态
    for t in range(T - 2, -1, -1):
        # 获取t+1时刻对应最优下标的值
        max_delta_index = pre_index[t + 1][max_delta_index]
        # 赋值
        decode[t] = max_delta_index

    return delta, decode, max_prob


if __name__ == '__main__':
    pi = np.array([0.2, 0.5, 0.3])
    A = np.array([
        [0.5, 0.4, 0.1],
        [0.2, 0.2, 0.6],
        [0.2, 0.5, 0.3]
    ])
    B = np.array([
        [0.4, 0.6],
        [0.8, 0.2],
        [0.5, 0.5]
    ])
    # 对A\B\pi进行log转换、
    pi = np.log(pi)
    A = np.log(A)
    B = np.log(B)
    n = np.shape(A)[0]
    Q = [0, 1, 0, 0, 1]

    print("测试前向概率计算....................")
    alpha = calc_alpha(pi, A, B, Q)
    print(alpha)
    print(np.exp(alpha))

    # 序列Q可能出现的概率值为
    p = np.exp(log_sum_exp(alpha[-1]))
    print("序列{}出现的可能性为:{}".format(Q, p))

    print("测试后向概率计算....................")
    beta = calc_beta(pi, A, B, Q)
    print("计算出来的后向概率矩阵beta为:")
    print(beta)
    print(np.exp(beta))

    # 计算序列Q出现的可能性
    p = np.zeros(len(A))
    for i in range(len(A)):
        p[i] = pi[i] + B[i][Q[0]] + beta[0][i]
    print("序列{}出现的可能性为:{}".format(Q, np.exp(log_sum_exp(p))))

    print("测试gamma概率计算....................")
    gamma = calc_gamma(alpha, beta)
    print(gamma)
    print(np.exp(gamma))

    print("viterbi算法应用.....................")
    delta, state_seq, max_prob = viterbi(pi, A, B, Q)
    print("最终结果为:", end='')
    print(state_seq)
    print(np.exp(max_prob))
    state = ['盒子1', '盒子2', '盒子3']
    for i in state_seq:
        print(state[i], end='\t')
