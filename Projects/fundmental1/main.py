import torch
from torch import nn
from d2l import torch as d2l


def corr2d(X, K):
    """
    计算二维互相关运算
    :param X: 输入
    :param K: 核矩阵
    :return:
    """
    h, w = K.shape
    # 输出
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # X [i~i+h,j~j+w] 与 K 做点积运算求和
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X = torch.tensor(
    [[0.0, 1.0, 2.0],
     [3.0, 4.0, 5.0],
     [6.0, 7.0, 8.0]]
)
K = torch.tensor(
    [[0.0, 1.0],
     [2.0, 3.0]]
)

corr2d(X, K)


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


"""
简单应用 检测图像中不同颜色的边缘
通过找到像素变化的位置 来检测图像中不同颜色的边缘
"""
# 首先构造一个6x8像素的黑白图像
# 中间四列为黑色0    其余像素为白色1
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)
# 构造一个高度为1 宽度为2 的卷积核
# 当进行互相关运算时 如果水平相邻的两元素相同 则输出为0 否则输出为非0
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
print(Y)

"""
学习由X生成Y的卷积核
给定输入X   输出Y     通过X,Y 去学习得到K
即通过输入 输出 去学习卷积核
"""

# 构造一个二维卷积层
# 它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

# 其实是一个 梯度下降
for i in range(10):
    Y_hat = conv2d(X)
    # loss
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')

# 通过10次迭代误差降到足够低后 查看一下卷积核的权重张量
# 迭代的次数越多最终的值越准确 越来越接近 [1,-1]
print(conv2d.weight.data.reshape((1, 2)))
