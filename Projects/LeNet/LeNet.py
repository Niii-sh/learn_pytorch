import torch
# pytorch库中 提供的网络模型 nn.Linear , nn.Conv2d, BatchNorm, Loss Functions
import torch.nn as nn

"""
LeNet architecture:
1x32x32 Input 
    -> (5x5),s=1,p=0 -> avg pool s=2,p=0->(5x5),s=1,p=0 -> avg pool s=2,p=0
    -> Conv 5x5 to 120  channels x Linear 84 x Linear 10
"""


class Reshape(torch.nn.Module):
    """
        将输入转换为批量数不变 通道数为1   28x28
    """

    def forward(self, x):
        return x.view(-1, 1, 28, 28)


net = torch.nn.Sequential(
    Reshape(),
    # 原始输入为32x32 但当前输入为28x28 所以padding=2 将两边补齐
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=(2, 2)),
    # 得到非线性 当时还没有ReLU 用的都是sigmoid
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
    # 卷积层出来为4维 最后将其转换1维的向量
    nn.Flatten(),
    # 最后一层 高和宽在池化后变为5x5 所以最后一层的输出是 16 x 5 x 5
    nn.Linear(16 * 5 * 5, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10)
)

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
# 由于是用Sequential建立 通过这种方法可以获得每层的输出
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
