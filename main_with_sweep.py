import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
from src.model import LinearModel
from src.data import dataset
from src.runner import train, test
from src.viz import loss_visualize, acc_visualize
import wandb
import os
import time
#os.environ["WANDB_MODE"] = "offline"

######################### Hyper-parameters #########################
defaults = dict(
    hidden_size = [256,128],
    num_epochs = 3000,
    learning_rate = 0.01,
    batch_size = 64
)
try:
    wandb.init(settings=wandb.Settings(start_method='fork'), config=defaults,project="696ds_deepmind", entity="696ds_deepmind")
except:
    wandb.init(config = defaults, projects = "696ds_deepmind", entity="696ds_deepmind")

config = wandb.config
hidden_size = config.hidden_size
out_size = 1
num_epochs = config.num_epochs
learning_rate = config.learning_rate
seed = config.seed
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
batch_size = config.batch_size


# modify this variable as required
with_clusters = True

if with_clusters:
    input_cols = ["x_1", "x_2", "cluster"]
    input_size = 3
else:
    input_cols = ["x_1", "x_2"]
    input_size = 2


########### Loading the dataset #############################
data_normal = pd.read_csv("data_normal.csv")

num_clusters = data_normal.cluster.nunique()

X_train, X_test, y_train, y_test = train_test_split(np.array(data_normal[input_cols]), 
                           np.array(data_normal["y"]), test_size=0.3)


trainset = dataset(torch.tensor(X_train,dtype=torch.float32).to(device), \
					torch.tensor(y_train,dtype=torch.float32).to(device))
testset = dataset(torch.tensor(X_test,dtype=torch.float32).to(device), \
					torch.tensor(y_test,dtype=torch.float32).to(device))

#DataLoader
trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
valloader = DataLoader(testset,batch_size=batch_size,shuffle=True)

# model definition
model = LinearModel(input_size, hidden_size, out_size, with_clusters = with_clusters, num_clusters = num_clusters, embed_dim=6).to(device)
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start = time.time()
model, tr_loss, tr_acc, val_acc = train(model, trainloader, valloader, \
						optimizer, num_epochs, criterion)
end = time.time()
torch.save(model, "model_{}.pt".format("_".join(map(str,hidden_size))))
loss_visualize(tr_loss, "Loss vs iteration", with_clusters = with_clusters)
acc_visualize([tr_acc, val_acc], \
				["training accuracy", "validation accuracy"], \
				"Accuracy vs epochs", with_clusters = with_clusters)
best_val_acc = max(val_acc)
wandb.log({"val_accuracy": best_val_acc})
wandb.log({"train_loss": tr_loss, "train_acc": max(tr_acc)})
wandb.log({"time": end - start})

