import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, with_clusters = False, num_clusters = 1, embed_dim=5):
        super(LinearModel, self).__init__()
        self.with_clusters = with_clusters
        if self.with_clusters == True:
            self.num_clusters = num_clusters
            self.embeds = nn.Embedding(num_embeddings=self.num_clusters, embedding_dim=embed_dim)
            input_size = input_size + embed_dim - 1
    
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], 1)
        
                           
    def get_weights(self):
        return self.weight
    
    def forward(self,x):
        if self.with_clusters == True:
            cluster_ids = x[:,-1]
            embeds_tensor = self.embeds(cluster_ids.long())
            x = x[:,:-1]
            x = torch.cat((x, embeds_tensor),dim=1)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out
    

