"""
多输入多输出通道
"""
import torch
from d2l import torch as d2l


def corr2d_multi_in(X, K):
    """
    多输入通道实现互相关运算
    :param X: 输入    3d
    :param K: 卷积核   3d
    对每个通道执行互相关操作 然后将结果相加
    :return:
    """
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

# tensor([[ 56.,  72.],
#           [104., 120.]])
# 3d 卷积核输出自然是3d的
print(corr2d_multi_in(X, K))


def corr2d_multi_in_out(X, K):
    """
    多输出通道实现互相关运算
    :param X: 输入    3d
    :param K: 卷积核  4d
    :return:
    """
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


# stack 0 代表在第0维 将K,K+1,K+2 堆叠形成新的矩阵
K = torch.stack((K, K + 1, K + 2), 0)
#   torch.Size([3, 2, 2, 2])
#   3个卷积核 每个卷积核2个通道 每个通道2*2的矩阵
print(K.shape)
