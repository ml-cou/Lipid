import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNPredictor(torch.nn.Module):
    def __init__(self, input_features, hidden_features):
        super(GCNPredictor, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_features)  # First GCN layer
        self.conv2 = GCNConv(hidden_features, hidden_features)  # Second GCN layer
        self.fc = torch.nn.Linear(hidden_features, 1)  # Linear layer for output

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))  # Apply first GCN layer and ReLU activation
        x = F.relu(self.conv2(x, edge_index))  # Apply second GCN layer and ReLU activation
        x = global_mean_pool(x, data.batch)  # Global mean pooling
        x = self.fc(x)  # Apply final linear layer
        return x.squeeze()  # Ensure output is 1D
