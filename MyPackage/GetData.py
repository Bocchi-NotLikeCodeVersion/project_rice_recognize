
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
# import os
# path1=os.path.dirname(__file__)
# train_path=os.path.join(path1,"assets/rice/train")
# train_path=os.path.relpath(train_path)
# val_path=os.path.join(path1,"assets/rice/val")
# val_path=os.path.relpath(val_path)
class GetRiceData():
    def __init__(self, path_train=None, path_val=None):
        # 初始化训练集路径
        self.path_train = path_train
        # 初始化验证集路径
        self.path_val = path_val
        pass
    def get_train_batch(self, batch_size):
        if not self.path_train:
            raise ValueError("No training data path.")
        # 创建ImageFolder数据集
        data_train = ImageFolder(
            root=self.path_train,  # 指定训练数据的根目录
            transform=transforms.Compose([  # 定义数据预处理流程
            # 调整图像大小
            transforms.Resize((128, 128)),
            # 将图像转换为灰度图
            transforms.Grayscale(num_output_channels=1),
            # 对图像进行亮度、对比度、饱和度和色调的随机调整
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # 将图像转换为Tensor格式
            transforms.ToTensor()
            ]))
        # print(data_train.classes)
        # 创建数据加载器
        data_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)  # 设置批量大小和是否打乱数据
        return data_loader
    def get_val_batch(self, batch_size=32):
        if not self.path_val:
            raise ValueError("No validation data path.")
        # 初始化ImageFolder类，用于加载验证数据集
        data_val = ImageFolder(
            # 设置验证数据集路径
            root=self.path_val,
            # 设置数据转换流程
            transform=transforms.Compose([
                transforms.Resize((128, 128)),
                # 将图像转换为灰度图
                transforms.Grayscale(num_output_channels=1),
                # 对图像进行随机亮度、对比度、饱和度和色调调整
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # 将图像转换为张量
                transforms.ToTensor()
            ]))
        
        # 使用DataLoader加载数据，设置批处理大小
        data_loader = DataLoader(data_val, batch_size=batch_size)
        return data_loader
    
if __name__=="__main__":
    # data1=GetRiceData(path_val=val_path)
    # data1.get_val_batch(16)
    pass