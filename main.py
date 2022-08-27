import train
import data
import train
import torch
import torch.nn as nn
from torch import optim
from data import Get_data_loader
from model import ViT


train_dl, val_dl=Get_data_loader(batch_size=32)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=ViT().to(device)
# define the loss function, optimizer and lr_scheduler
loss_fn = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# from torch.optim.lr_scheduler import ReduceLROnPlateau
# lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)


# params_train = {
#     'num_epochs':30,
#     'optimizer':opt,
#     'loss_func':loss_func,
#     'train_dl':train_dl,
#     'val_dl':val_dl,
#     'lr_scheduler':lr_scheduler,

# }
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 예측(prediction)과 손실(loss) 계산
        X=X.to(device)
        y=y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X,y=X.to(device),y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dl, model, loss_fn, optimizer)
    test_loop(val_dl, model, loss_fn)
print("Done!")