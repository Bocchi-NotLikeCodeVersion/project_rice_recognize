
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
# import os
# path1=os.path.dirname(__file__)
# train_path=os.path.join(path1,"assets/rice/train")
# train_path=os.path.relpath(train_path)
# val_path=os.path.join(path1,"assets/rice/val")
# val_path=os.path.relpath(val_path)
class GetRiseData():
    def __init__(self,path_train,path_val):
        self.path_train=path_train
        self.path_val=path_val
        pass
    def get_train_batch(self,batch_size):
        data_train=ImageFolder(
            root=self.path_train,
            transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor()
            ]))
        data_loader=DataLoader(data_train,batch_size=batch_size,shuffle=True)
        return data_loader
    def get_val_batch(self,batch_size):
        data_val=ImageFolder(
            root=self.path_val,
            transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor()
            ]))
        data_loader=DataLoader(data_val,batch_size=batch_size)
        return data_loader
    
if __name__=="__main__":
    # data1=GetRiseData(val_path)
    # data1.get_val_batch(16)
    pass