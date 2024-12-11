import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def score(model,data_loader):
    model = model.to(device)
    model.eval()
    score = 0
    epoch=0
    for inputs,labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        y_pred = torch.argmax(outputs, dim=1)
        score+=(y_pred == labels).sum().item() / len(labels)
        epoch+=1
    return score/epoch
    