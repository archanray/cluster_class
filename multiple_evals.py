import pickle
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def load_cifar_results(model=20, type_=True):
    y_preds = []
    all_y_preds = []
    directory = "./saved_outputs/"
    files_with_cluster = directory+"c*"
    files_without_cluster = directory+"nc*"
    return y_preds, all_y_preds

def visualize_matrix(mat, w, mode="normal", data_type="cluster_class"):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()
    ax.matshow(mat, cmap='ocean')

    for i in range(len(mat)):
        for j in range(len(mat)):
            c = mat[j, i]
            ax.text(i, j, "{:.4f}".format(c), va='center', ha='center')

    plt.savefig("figures/"+data_type+"_"+str(w)+"_"+mode+".pdf")

def consistency(all_ops):
    prediction_stds = np.std(outputs, axis=1)
    return np.mean(prediction_stds)

def consistency_nicks(all_ops, y_true):
    y_true = np.array(y_true)
    y_true = y_true[:, np.newaxis]
    correctness = (all_ops == y_true).astype(int)
    correctness = np.sum(correctness, axis=1)
    return np.mean(correctness)

def consistency_across_all(all_ops):
    max_val = all_ops.shape[1]
    target_vals = [0, max_val]
    target_vec_low = target_vals[0]*np.ones((all_ops.shape[0], 1))
    target_vec_high = target_vals[1]*np.ones((all_ops.shape[0], 1))
    total_ops = np.sum(all_ops, axis=1)
    total_ops = total_ops[:, np.newaxis]
    v_low = (total_ops == target_vec_low).astype(int)
    v_high = (total_ops == target_vec_high).astype(int)
    v_total = v_low+v_high
    v_total[v_total < 1] = 0
    v_total[v_total > 1] = 1
    v_total = np.sum(v_total.astype(int)) / all_ops.shape[0]
    return v_total

def consitency_pairwise(all_ops):
    n = all_ops.shape[0]
    consistency_matrix = np.zeros((all_ops.shape[1], all_ops.shape[1]))
    consistency_array = []
    for i in range(len(consistency_matrix)):
        for j in range(i+1, len(consistency_matrix)):
            consistency_matrix[i,j] = np.sum((all_ops[:,i] == all_ops[:,j]).astype(int)) / n
            consistency_array.append(consistency_matrix[i,j])

    scores_mean = np.mean(consistency_array)
    scores_std = np.std(consistency_array)
    return scores_mean, scores_std, consistency_matrix

def consistency_pairwise_with_true(all_ops, true_y):
    n = all_ops.shape[0]
    consistency_matrix = np.zeros((all_ops.shape[1], all_ops.shape[1]))
    consistency_array = []
    for i in range(len(consistency_matrix)):
        for j in range(i+1, len(consistency_matrix)):
            f1 = all_ops[:,i] != true_y
            f2 = all_ops[:,j] != true_y
            # indices from 1
            id1 = np.where(f1 == True)[0]
            id2 = np.where(f2 == True)[0]
            union_id = set.union(set(id1),set(id2))
            union_id = list(union_id)
            f1 = f1[union_id]
            f2 = f2[union_id]
            consistency_matrix[i,j] = np.sum((f1 == f2).astype(int)) / len(union_id)
            consistency_array.append(consistency_matrix[i,j])

    scores_mean = np.mean(consistency_array)
    scores_std = np.std(consistency_array)
    return scores_mean, scores_std, consistency_matrix

with_clusters = [False, True]
data_type = "cluster_class"
res_model = 20

for w in with_clusters:
    print(w)
    if data_type == "cluster_class":
        filename = "saved_outputs/"+str(w)+".pickle"
        accuracies = []

        with open(filename, "rb") as f:
            [y_true, all_y_preds] = pickle.load(f)
    
        for i in range(len(all_y_preds)):
            accuracies.append(accuracy_score(y_true, all_y_preds[i]))

    if data_type == "CIFAR100":
        y_true, all_y_preds = load_cifar_results(model=res_model, type_=with_clusters)
        pass

    num_samples = len(y_true)
    num_models = len(all_y_preds)
    outputs = np.zeros((num_samples, num_models))

    for i in range(num_models):
        outputs[:,i] = all_y_preds[i]
    print(np.mean(accuracies), np.std(accuracies))

    #predictions_mean = np.mean(outputs, axis=1)
    #predictions_stds = np.std(outputs, axis=1)

    print("Churn? maybe:", consistency(outputs))
    print("Churn? maybe:", consistency_nicks(outputs, y_true))
    print("Churn? maybe:", consistency_across_all(outputs))

    [cs_mean, cs_std, mat] = consitency_pairwise(outputs)
    print("Churn? maybe:", cs_mean, cs_std)

    visualize_matrix(mat, w)
    
    [cs_mean, cs_std, mat] = consistency_pairwise_with_true(outputs, y_true)
    print("Churn? mistakes:", cs_mean, cs_std)
    visualize_matrix(mat, w, mode="mistakes")
    
















