import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data_normal = pd.read_csv("data_normal.csv")
print(len(data_normal))

X_train, X_test, y_train, y_test = train_test_split(np.array(data_normal[["x_1", "x_2"]]), 
                           np.array(data_normal["y"]), test_size=0.3)

data_0 = X_test[y_test==0,:]
data_1 = X_test[y_test==1,:]

label_0 = y_test[y_test==0]
label_1 = y_test[y_test==1]

plt.scatter(data_0[:,0], data_0[:,1])
plt.scatter(data_1[:,0], data_1[:,1])
plt.show()
