"""
常见的transform

transform的总结:
看官方文档
1. 关注输入和输出类型
2. 关注方法需要的参数
不知道返回类型时 可考虑 print(type())
"""
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

writer = SummaryWriter("logs_trans")
img = Image.open("images/test.jpg")
print(img)

# ToTensor  读入图片
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

# Normalize 重新规划图片相关信息
"""
规划的计算方式: 
    输出值  = (输入值 - 平均值) / 标准值
``output[channel] = (input[channel] - mean[channel]) / std[channel]``
"""
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([2,4,3],[1,2,2])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
# 加入规划后的图片 这里主要是色调会发生变化
writer.add_image("Normalize",img_norm)

# Resize 改变图像的大小
print(img.size)
trans_resize = transforms.Resize((512,512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> img_resize (tensor数据类型)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)
print(img_resize)


# Compose() - resize - 2
"""
Compose()中的参数需要是一个列表
Python中 列表的表示形式为 [数据1, 数据2, ...]
在Compose中 数据需要是 transforms类型
所以得到 Compose([transform参数1,transform参数2, ...])
"""
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)


# RandomCrop 随机裁剪
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)

writer.close()

