# 定义模型 选择vgg11模型
import os
import torch
from torchvision.models import resnet34,ResNet34_Weights
from torch import nn
path1=os.path.dirname(__file__)
model_path=os.path.join(path1,"assets/pth/vgg11-8a719046.pth")
model_path=os.path.relpath(model_path)

class CustomResnet34():
    def __init__(self):
        net1 = resnet34()
        # 修改网络的第一层卷积层，输入通道数从3改为1
        net1.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        net1.fc = nn.Linear(in_features=512, out_features=5, bias=True)
        net1.add_module('SoftMax', nn.Softmax(dim=1))
        # 将修改后的网络赋值给self.net1
        self.net1 = net1
        # print(net1)
    def resnet34_has_trained(self,path):
        # 将net1赋值给net2
        net2 = self.net1
        # 加载预训练权重
        pre_static_dict = torch.load(path, weights_only=True)
        # print(pre_static_dict.keys())

        # 从预训练权重中移除不需要的层权重
        
        pre_static_dict.pop("conv1.weight")
        
        

        pre_static_dict.pop("fc.weight")
        
        pre_static_dict.pop("fc.bias")

        # 获取net2的当前状态字典
        my_state_dict = net2.state_dict()

        # 更新net2的状态字典，使用预训练权重
        my_state_dict.update(pre_static_dict)

        # 将更新后的状态字典加载到net2中
        net2.load_state_dict(my_state_dict)

        # 打印net2的模型结构
        # print(net2)

        return net2
    def get_common_custom_resnet34(self):
        return self.net1

if __name__=="__main__":
    # r1=CustomResnet32()
    # net1=r1.resnet34_has_trained(r"assets\pth\resnet34-b627a593.pth")
    # print(net1)
    pass