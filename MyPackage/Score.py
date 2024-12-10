import torch
def score(model,data_loader):
    model.eval()
    score = 0
    epoch=0
    for inputs,labels in data_loader:
        outputs = model(inputs)
        y_pred = torch.argmax(outputs, dim=1)
        score+=(y_pred == labels).sum().item() / len(labels)
        epoch+=1
    return score/epoch
    