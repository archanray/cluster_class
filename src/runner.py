import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(model, train_loader, val_loader, optimizer, num_epochs, criterion):
    model.train()
    y_true = []
    y_pred = []

    loss_all_epochs = []
    tr_acc_all_epochs = []
    val_acc_all_epochs = []
    for epoch_ in tqdm(range(num_epochs)):
        total_loss = []

        for i in train_loader:
            data, target = i
            # plt.scatter(data[target==0,0].cpu(), data[target==0,1].cpu())
            # plt.scatter(data[target==1,0].cpu(), data[target==1,1].cpu())
            # plt.show()

            # zero the parameter gradients
            optimizer.zero_grad()
            #FORWARD PASS
            output = model(data.to(device))
            loss = criterion(output, target.to(device).unsqueeze(1)) 
            #BACKWARD AND OPTIMIZE
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
        if (epoch_ % 100) == 0:
            print("Loss: ", np.mean(total_loss))

        # PREDICTIONS 
        val_acc_all_epochs.append(test(model, val_loader))
        tr_acc_all_epochs.append(test(model, train_loader))
        
        # save the losses
        loss_all_epochs += total_loss

    return model, loss_all_epochs, tr_acc_all_epochs, val_acc_all_epochs


#TESTING THE MODEL
def test(model, test_loader, mode="eval only"):
    #model in eval mode skips Dropout etc
    model.eval()
    y_true = []
    y_pred = []
    
    sigmoid = nn.Sigmoid()

    # set the requires_grad flag to false as we are in the test mode
    with torch.no_grad():
        for i in test_loader:
            #LOAD THE DATA IN A BATCH
            data,target = i
            
            # the model on the data
            output = model(data.to(device))
            output = sigmoid(output)
                       
            #PREDICTIONS
            pred = np.round(output.cpu().numpy())
            target = target.cpu().numpy()
            y_true.extend(target.tolist()) 
            y_pred.extend(pred.reshape(-1).tolist())
    if mode == "eval only":
        return accuracy_score(y_true,y_pred)
    if mode == "eval labels":
        return accuracy_score(y_true, y_pred), y_true, y_pred
    
