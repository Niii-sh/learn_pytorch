"""
关于add_image()的使用
"""
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("log_img")
image_path = "data/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
# 将图片转化为 numpy.array型 才能正确的传入 add_image() 函数
img_array = np.array(img_PIL)
"""
self, 
tag, img_tensor, global_step=None, walltime=None, dataformats="CHW"

从PIL 到 numpy 需要在add_image()中 指定shape中 每一个数字/维表示的含义
"""
writer.add_image("test", img_array, 1, dataformats='HWC')

# 这样可以使得两张出现在不同的title下  否则两张图片都会出现在test下
image_path2 = "dataset/train/ants/5650366_e22b7e1065.jpg"
writer.add_image("test1",np.array(Image.open(image_path2)),2,dataformats='HWC')

writer.close()