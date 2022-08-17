import matplotlib.pyplot as plt
from datetime import datetime

def loss_visualize(train_loss, title, with_clusters = False, model="cluster_center"):
	now = datetime.now()
	path = "figures/baseline/"
	if with_clusters == True:
		path = "figures/with_clusters/"
	plt.clf()
	plt.plot(train_loss)
	plt.xlabel("Iterations")
	plt.ylabel("Loss")
	plt.title(title)
	plt.savefig(path + model+"_"+"train_loss_"+str(now)+".pdf")

	return None

def acc_visualize(accuracies, labels, title, with_clusters = False, model="cluster_center"):
	now = datetime.now()
	path = "figures/baseline/"
	if with_clusters == True:
		path = "figures/with_clusters/"
	plt.clf()
	plt.title(title)
	for i in range(len(accuracies)):
		plt.plot(accuracies[i], label=labels[i])
		plt.xlabel("Epochs")
		plt.ylabel("Accuracy")
		plt.legend(loc="upper right")
		plt.savefig(path + model+"_"+"train_accuracies_"+str(now)+".pdf")			
	return None
