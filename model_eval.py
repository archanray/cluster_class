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
import pickle
from src.runner import test
import os
from glob import glob
from sklearn.metrics import accuracy_score

######################### Hyper-parameters #########################
#hidden_size = [2048,1024] #[40,20]
hidden_size = [256,128]
out_size = 1
print("_".join(map(str,hidden_size)))
num_epochs = 200
learning_rate = 0.0003
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
batch_size = 256

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

if with_clusters == False:
    with open("./save_parted/datasplits_wo_clusters.pickle", "rb") as f:
        [X_train, X_test, y_train, y_test] = pickle.load(f)

if with_clusters == True:
    with open("./save_parted/datasplits_wt_clusters.pickle", "rb") as f:
        [X_train, X_test, y_train, y_test] = pickle.load(f)

num_clusters = data_normal.cluster.nunique()

testset = dataset(torch.tensor(X_test,dtype=torch.float32).to(device), \
                                        torch.tensor(y_test,dtype=torch.float32).to(device))

#DataLoader
valloader = DataLoader(testset,batch_size=batch_size,shuffle=True)

# load the model
## instantiate models for all files
all_model_files = glob("saved_models/"+str(with_clusters)+"*")
models = []
for i in range(len(all_model_files)):
    models.append(LinearModel(input_size, hidden_size, out_size, \
                    with_clusters = with_clusters, \
                    num_clusters = num_clusters, \
                    embed_dim=6).to(device))

## load state dicts for all models and prep for evaluation
for i in range(len(all_model_files)):
    model_file = all_model_files[i]
    models[i].load_state_dict(torch.load(model_file))
    models[i].eval()

y_true = []
y_preds = []

sigmoid = nn.Sigmoid()

with torch.no_grad():
    for i in valloader:
        #LOAD THE DATA IN A BATCH
        data,target = i
        # run the models on the data
        local_preds = []
        for i in range(len(models)):
            output = models[i](data.to(device))
            output = sigmoid(output)
            pred = np.round(output.cpu().numpy())
            pred = pred.reshape(-1).tolist()
            local_preds.append(pred)
        y_preds.append(local_preds)
        target = target.cpu().numpy()
        y_true.extend(target.tolist())

formatted_y_preds = []
for i in range(len(models)):
    formatted_y_preds.append([])

for i in range(len(models)):
    count = 0
    for j in valloader:
        formatted_y_preds[i].extend(y_preds[count][i])
        count += 1

"""
for i in range(len(models)):
    print(accuracy_score(y_true, formatted_y_preds[i]))
"""

filename = os.path.join("saved_outputs", str(with_clusters)+".pickle")
with open(filename, "wb") as f:
    pickle.dump([y_true, formatted_y_preds], f, protocol=pickle.HIGHEST_PROTOCOL)



























