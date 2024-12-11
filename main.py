import os
import torch
import time
from PIL import Image
from MyPackage.CNNNet import CustomResnet34
from MyPackage.GetData import GetRiceData
from MyPackage.Train import Train
from MyPackage.Predict import predict
from MyPackage.Score import score
from torchvision import transforms
path1=os.path.dirname(__file__)
train_path=os.path.join(path1,"assets/rice/train")
train_path=os.path.relpath(train_path)
val_path=os.path.join(path1,"assets/rice/val")
val_path=os.path.relpath(val_path)

model_path=os.path.join(path1,"assets/pth/resnet34-b627a593.pth")
model_path=os.path.relpath(model_path)

model_path2=os.path.join(path1,"assets/pth/resnet34-last-20241211145447.pth")
model_path2=os.path.relpath(model_path2)

now_time=time.strftime(r"%Y%m%d%H%M%S",time.localtime())
model_path_last=os.path.join(path1,f"assets/pth/resnet34-last-{now_time}.pth")
model_path_last=os.path.relpath(model_path_last)

def train_and_save_model(batch_size=16,epoch=1):
    data_obj=GetRiceData(path_train=train_path)
    data_lodar=data_obj.get_train_batch(batch_size) 
    net1_obj=CustomResnet34()
    net1=net1_obj.resnet34_has_trained(model_path)
    train_obj=Train(net1,state_dict_path=model_path2)
    model=train_obj.begin_train(data_lodar,epoch)
    train_obj.save_model(model_path_last)
    pass
    
def inference(img_path):
    model_obj=CustomResnet34()
    model=model_obj.get_common_custom_resnet34()
    model.load_state_dict(torch.load(model_path2,weights_only=True))
    res=predict(model,img_path)
    print(res)
    
def evaluate():
    data_obj=GetRiceData(path_val=val_path)
    data_lodar=data_obj.get_val_batch() 
    model_obj=CustomResnet34()
    model=model_obj.get_common_custom_resnet34()
    model.load_state_dict(torch.load(model_path2,weights_only=True))
    score_value=score(model,data_lodar)
    print(score_value)
if __name__=="__main__":
    # train_and_save_model(32,100)
    # inference(r"assets\rice\val\Jasmine\Jasmine (1).jpg")
    evaluate()
    pass