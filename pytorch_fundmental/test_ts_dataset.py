"""
dataset 和 transform的联合使用
"""

import torchvision
import ssl

# 全局取消证书验证  否则会报错
# 这里是个坑 很多需要联网下载报错都和这个ssl有关系
from torch.utils.tensorboard import SummaryWriter

ssl._create_default_https_context = ssl._create_unverified_context

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

#  root 数据集存放的位置
# CIFAR10为一个数据集 比较小 适合学习使用
# 参数 transfrom = dataset_transform  将 CIFAR10 中数据类型转换成 Tensor类型 从而可以使用 tensorboard等工具进行操作
train_set = torchvision.datasets.CIFAR10(root="./dataset2",train=True,transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset2",train=False,transform=dataset_transform,download=True)

print(test_set[0])
print(test_set.classes)

# # 获取第0个 数据的图片以及target
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# # 展示图片
# img.show()

# print(test_set[0])

writer = SummaryWriter("dataset_ts")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()