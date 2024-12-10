# 定义模型 选择vgg11模型
import os
import torch
from torchvision.models import vgg11,VGG11_Weights
from torch import nn
path1=os.path.dirname(__file__)
model_path=os.path.join(path1,"assets/pth/vgg11-8a719046.pth")
model_path=os.path.relpath(model_path)

class CustomVGG11():
    def __init__(self):
        net1=vgg11()
        net1.features[0]=nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        net1.classifier.append(nn.Linear(in_features=1000, out_features=10, bias=True))
        net1.classifier.append(nn.Softmax())
        self.net1=net1
        print(net1)
    def vgg11_has_trained(self,path):
        net2=self.net1
        pre_static_dict=torch.load(path,weights_only=True)
        pre_static_dict.pop("features.0.weights")
        pre_static_dict.pop("features.0.bias")
        
        pass


if __name__=="__main__":
    v1=CustomVGG11()
    # v1.vgg11_has_trained(model_path)
    pass