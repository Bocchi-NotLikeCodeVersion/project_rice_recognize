import torch
import wandb
import time
import torch.optim as optim
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def init_wandb(lr,epoch):
    wandb.init(
        # set the wandb project where this run will be logged
        project="rice", # 工程名

        # track hyperparameters and run metadata
        config={
        "learning_rate": lr, # 模型的学习率
        "architecture": "VGG11", # 模型是CNN
        "dataset": "Custom", # 数据集名称
        "epochs": epoch,# 轮次
        }
    )

class Train():
    def __init__(self,model,optimizer=None,state_dict_path=None):
        if state_dict_path is not None:
            model.load_state_dict(torch.load(state_dict_path,weights_only=True))
        self.model = model.to(device)
        if optimizer is None:
            self.optimizer = self.init_optimizer()
        else:
            self.optimizer = optimizer
        self.loss_fn = self.init_loss()
    def begin_train(self, data_loader,epoch,visualization=True):
        if visualization:
            init_wandb(lr=self.optimizer.param_groups[0]['lr'],epoch=epoch)
            for i in range(epoch):
                start_time = time.time()
                for inputs, labels in data_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs,labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if i % 5 == 0:
                        y_pred = torch.argmax(outputs, dim=1)
                        wandb.log({'loss': loss,'accuracy': (y_pred == labels).sum().item() / len(labels)})
                end_time = time.time()
                print(f"epoch {i}: time:{end_time - start_time} loss:{loss:.4f}")
            wandb.watch(self.model, log="all", log_graph=True)
            wandb.finish()
                    
            return self.model
        
        for i in range(epoch):
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs,labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        print("Training Finished!")
        return self.model
    
    def init_optimizer(self,lr=1e-5):
       optimizer=optim.Adam(self.model.parameters(), lr=lr)
       return optimizer
    def init_loss(self):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn
    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
    