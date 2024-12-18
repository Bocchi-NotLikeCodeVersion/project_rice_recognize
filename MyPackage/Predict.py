import torch
from torchvision import transforms
from PIL import Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def predict(model, path):
    model = model.to(device)
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
    model.eval()
    data = transform(data)
    data = data.unsqueeze(0)
    data = data.to(device)
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1)
        res=res_list[pred]
        return res
