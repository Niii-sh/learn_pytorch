"""
transform
"""
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

"""
python的用法 -> tensor数据类型
通过 transforns.ToTensor去看两个问题
1.transform的使用
2.为什么需要Tensor 数据类型
"""
img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs_trans")

# transform 的使用 将图片转化为 tensor类型
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img",tensor_img)
writer.close()
