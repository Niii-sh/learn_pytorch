"""
linear layer
线性层的应用
"""
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataLoader = DataLoader(dataset, batch_size=64,drop_last=True)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


tudui = Tudui()

for data in dataLoader:
    imgs, targets = data
    print(imgs.shape)
    # 展平数据
    output =  torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)
