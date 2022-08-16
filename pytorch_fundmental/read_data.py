from torch.utils.data import Dataset
import cv2
# 用于处理图片信息
from PIL import Image
# 用于处理文件路径
import os


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        # 获取图片名称
        img_name = self.img_path[idx]
        # 构成图片的路径
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        # 读取图片
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)
# 整个数据集
train_dataset = ants_dataset + bees_dataset