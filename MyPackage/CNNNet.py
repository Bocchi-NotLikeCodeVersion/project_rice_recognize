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
        # 初始化VGG11网络
        net1 = vgg11()
        # 修改网络的第一层卷积层，输入通道数从3改为1
        net1.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 向分类器中添加一个新的全连接层，输出维度为10
        net1.classifier.append(nn.Linear(in_features=1000, out_features=5, bias=True))
        # 向分类器中添加Softmax层
        net1.classifier.append(nn.Softmax(dim=1))
        # 将修改后的网络赋值给self.net1
        self.net1 = net1
        # print(net1)
    def vgg11_has_trained(self,path):
        # 将net1赋值给net2
        net2 = self.net1

        # 加载预训练权重
        pre_static_dict = torch.load(path, weights_only=True)
        # print(pre_static_dict.keys())

        # 从预训练权重中移除不需要的层权重
        # 移除第一个卷积层的权重
        pre_static_dict.pop("features.0.weight")
        # 移除第一个卷积层的偏置
        pre_static_dict.pop("features.0.bias")

        # 获取net2的当前状态字典
        my_state_dict = net2.state_dict()

        # 更新net2的状态字典，使用预训练权重
        my_state_dict.update(pre_static_dict)

        # 将更新后的状态字典加载到net2中
        net2.load_state_dict(my_state_dict)

        # 打印net2的模型结构
        # print(net2)

        return net2
    def get_common_custom_vgg11(self):
        return self.net1

if __name__=="__main__":
    # v1=CustomVGG11()
    # v1.vgg11_has_trained(model_path)
    pass