# ONNX 可以把训练出来的模型给其他语言使用
import torch
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 使用onnx推理
class OnnxControls():
    def __init__(self,model,example,path):
        self.path=path
        self.model=model
        self.example=example
    def save_onnx(self):
        torch.onnx.export(self.model,self.example,self.path,  verbose=True,input_names=["input"],output_names=["output"])
    def onnx_pre(self,path): 
        res_list = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        data = Image.open(path).convert('RGB')
        transform=transforms.Compose([  # 定义数据预处理流程
                # 调整图像大小
                transforms.Resize((128, 128)),
                # 将图像转换为灰度图
                transforms.Grayscale(num_output_channels=1),
                # 将图像转换为Tensor格式
                transforms.ToTensor()
                ])
        data = transform(data)
        data = data.unsqueeze(0)
        data = data.to(device)
        data=data.numpy()
        # 推理
        sess = ort.InferenceSession(self.path)
        # 运行onnx模型
        res = sess.run(None, {"input": data})
        res = torch.argmax(torch.tensor(res))
        res=res_list[res]
        return res

    