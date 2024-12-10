import torch
import wandb
import torch.optim as optim
import torch.nn as nn
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
    def __init__(self,model,optimizer=None):
        self.model = model
        if optimizer is None:
            self.optimizer = self.init_optimizer()
        else:
            self.optimizer = optimizer
        self.loss_fn = self.init_loss()
    def begin_train(self, data_loader,epoch,visualization=True):
        if visualization:
            init_wandb(lr=self.optimizer.param_groups[0]['lr'],epoch=epoch)
            for i in range(epoch):
                for inputs, labels in data_loader:
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs,labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    y_pred = torch.argmax(outputs, dim=1)
                    wandb.log({'loss': loss,'accuracy': (y_pred == labels).sum().item() / len(labels)})
            wandb.watch(self.model, log="all", log_graph=True)
            wandb.finish()
                    
            return self.model
        
        for i in range(epoch):
            for inputs, labels in data_loader:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs,labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        print("Training Finished!")
        return self.model
    
    def init_optimizer(self,lr=1e-3):
       optimizer=optim.Adam(self.model.parameters(), lr=lr)
       return optimizer
    def init_loss(self):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn
    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
    