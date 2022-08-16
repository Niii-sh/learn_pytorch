import ssl

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ssl._create_default_https_context = ssl._create_unverified_context

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()

writer = SummaryWriter("../logs")

step = 0

for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    # 输入大小 torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # 输出大小 torch.Size([64, 6, 30, 30]) --> [xxx,3,30,30]
    # 将out 从6channel 转换为 3channel 否则无法正常加入
    # -1表示让 reshape 函数自动填入batch size
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step = step + 1
