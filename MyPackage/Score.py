import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def score(model,data_loader,visualization=False):
    if visualization:
        arr1=np.zeros((0,7))
        model = model.to(device)
        model.eval()
        score_value = 0
        epoch=0
        for inputs,labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            y_pred = torch.argmax(outputs, dim=1)
            score_value+=(y_pred == labels).sum().item() / len(labels)
            epoch+=1
            arr2=np.array([[outputs.detach().to("cpu").numpy()]])
            arr2=arr2[0][0]
            # print(arr2.shape)
            arr3=np.array([[y_pred.detach().to("cpu").numpy()]])
            arr3=arr3[0]
            arr4=np.array([[labels.detach().to("cpu").numpy()]])
            arr4=arr4[0]

            arr5=np.concatenate((arr3,arr4),axis=0).T
            arr6=np.concatenate((arr2,arr5),axis=1)
            arr7=np.concatenate((arr1,arr6),axis=0)
            arr1=arr7
        df1=pd.DataFrame(arr1)
        df1.columns=['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag',"y_pred","y_true"]
        # df1.to_csv('res.csv')
        pred_res=df1["y_pred"].values
        true_res=df1["y_true"].values
        
        matrix=confusion_matrix(true_res,pred_res)
        print(matrix)
        print(classification_report(true_res,pred_res))
        plt.matshow(matrix,cmap=plt.cm.Greens)
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                plt.annotate(
                    matrix[i,j],
                    xy=(j,i),
                    horizontalalignment="center",
                    verticalalignment="center",
                )
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.show()
        return score_value/epoch
    

    model = model.to(device)
    model.eval()
    score_value = 0
    epoch=0
    for inputs,labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        y_pred = torch.argmax(outputs, dim=1)
        score_value+=(y_pred == labels).sum().item() / len(labels)
        epoch+=1
        # print(epoch)
    return score_value/epoch
    