"""
关于dataloader的使用
"""
import torchvision

# 准备的测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset2",train=False,transform=torchvision.transforms.ToTensor())

"""
dataset : 要取的数据集 一般返回img和label
batch_size: 每次从dataset中取出多少数据进行打包
shuffle: 是否打乱数据 在每个epoch开始对数据进行重新排序
num_works: 加载数据的时候采用单进程还是多进程 默认设置为0 意为主进程进行加载
drop_last: 当数据集最后一批小于batch_size时 是否舍去最后一批数据集
"""
test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

# 测试数据集中第一张图片及target
img,target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
step = 0
for epoch in range(2):
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch),imgs,step)
        step = step + 1

writer.close()
