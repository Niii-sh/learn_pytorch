import torchvision
import os
# 这个数据集非公开 而且太大了不方便下载
# train_data = torchvision.datasets.ImageNet(
#     "../data_image_net",split='train',download=True,
#     transform=torchvision.transforms.ToTensor()
# )


vgg16_false = torchvision.models.vgg16()
vgg16_True = torchvision.models.vgg16()
print(vgg16_True)