"""
常见的transform
"""
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

writer = SummaryWriter("logs_trans")
img = Image.open("images/test.jpg")
print(img)

# ToTensor 的使用 读入图片
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

# Normalize 的使用
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

writer.close()

