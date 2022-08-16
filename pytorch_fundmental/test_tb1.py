"""
关于图像绘制的基本操作
"""

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
# writer.add_image()
# y = x
for i in range(100):
    """
    def add_scalar(
        self,
        tag,            # title
        scalar_value,   # y轴
        global_step=None, # x轴
        walltime=None,
        new_style=False,
        double_precision=False,
    ):
    
    然后要稍微注意下 生成图像会默认放在同一个目录下 
    所以不想出现图像叠层的情况 就是两种图像叠在一起 那么最好就是新建一个目录 将图像存储在新的目录下 或者就是原来的删除
    """
    writer.add_scalar("y=2x", 2 * i, i)

writer.close()
