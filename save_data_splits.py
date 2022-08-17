import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import os

# modify this variable as required
with_clusters = True

if with_clusters:
    input_cols = ["x_1", "x_2", "cluster"]
    input_size = 3
else:
    input_cols = ["x_1", "x_2"]
    input_size = 2


# load the dataset
data_normal = pd.read_csv("data_normal.csv")

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(np.array(data_normal[input_cols]),
                           np.array(data_normal["y"]), test_size=0.3)

# save to avoid thhese steps in future
## check if save folder exists
dir_path = "./save_parted"
Path(dir_path).mkdir(parents=True, exist_ok=True)
## save the pickle
filename = os.path.join(dir_path, "datasplits_wt_clusters.pickle")
with open(filename, "wb") as f:
    pickle.dump([X_train, X_test, y_train, y_test], f, protocol=pickle.HIGHEST_PROTOCOL)
