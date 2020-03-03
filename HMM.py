# -*- encoding:utf-8 -*-
"""
利用Python和numpy实现HMM
"""

import numpy as np

class HiddenMarkov:
    def forward(self, Q, V, A, B, O, PI):
        """
        前向算法
        :param Q: 状态有限集
        :param V: 序列内容集合
        :param A: 状态转移概率
        :param B: 状态转移时序列内容集合的分布概率
        :param O: 观测序列
        :param PI: 初始状态概率
        :return:
        """
        N = len(Q) # 可能存在的状态数量
        M = len(O) # 观测序列的大小

        alphas = np.zeros(N,M)
        T = M
        for t in range(T):
            indexOfO = V.index(O[t]) # 找出序列对应的索引
            for i in range(N):
                if t == 0:
                    alphas[i][t] = PI[t][i] * B[i][indexOfO]  # 初值，李航《统计学习方法》第10章 公式10.15
                else:
                    # 递推, 公式 10.16
                    alphas[i][t] = np.dot([alpha[t-1] for alpha in alphas], [a[i] for a in A]) * B[i][indexOfO]

        P = np.sum([alpha[M-1] for alpha in alphas])

    def backward(self, Q, V, A, B, O, PI):
        N = len(Q)
        M = len(O)
        betas = np.ones(N,M)
        for t in range(M-2, -1, -1):
            indexOfO = V.index(O[t+1])  # 找出序列对应的索引
            for i in range(N):
                betas[i][t] = np.dot(np.multiply(A[i], [b[indexOfO] for b in B]), [beta[t+1] for beta in betas])
                realT = t + 1
                realI = i + 1
        indexOfO = V.index(O[0])
        P = np.dot(np.multiply(PI, [b[indexOfO] for b in B]), [beta[0] for beta in betas])

    def viterbi(self, Q, V, A, B, O, PI):
        N = len(Q)
        M = len(O)
        deltas = np.zeros((N,M))
        psis = np.zeros((N,M))
        I = np.zeros((1,M))
        for t in range(M):
            realT = t + 1
            indexOfO = V.index(O[t])

            for i in range(N):
                realI = i+1
                if t == 0:
                    deltas[i][t] = PI[0][i] * B[i][indexOfO]
                    psis[i][t] = 0

                else:
                    deltas[i][t] = np.max(np.multiply([delta[t-1] for delta in deltas], [a[i] for a in A])) * B[i][indexOfO]
                    psis[i][t] = np.argmax(np.multiply([delta[t-1] for delta in deltas], [a[i] for a in A])) + 1
        I[0][M-1] = np.argmax([delta[M-1] for delta in deltas]) + 1
        for t in range(M -2, -1 , -1):
            I[0][t] = psis[int(I[0][t+1]) - 1][t+1]
