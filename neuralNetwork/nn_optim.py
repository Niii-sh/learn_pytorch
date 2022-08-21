"""
优化器
"""

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataLoader = DataLoader(dataset, batch_size=1)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 使用 Sequential 自定义神经网络的执行顺序
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


# 交叉熵
loss = nn.CrossEntropyLoss()

tudui = Tudui()
# 优化器
# lr 学习速率
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)

for epoch in range(20):
    # 每轮学习误差的总和
    running_loss = 0.0
    for data in dataLoader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)
        # 将优化器中的每一个参数的梯度清零
        optim.zero_grad()
        # 调用损失函数反向传播求出每一个节点的梯度
        result_loss.backward()
        # 然后再调用optim 对模型的每一个参数进行调优
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)
