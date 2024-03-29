import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class SAGEPredictor(torch.nn.Module):
    def __init__(self, input_features, hidden_features):
        super(SAGEPredictor, self).__init__()
        self.conv1 = SAGEConv(input_features, hidden_features, aggr='mean')  # First GraphSAGE layer
        self.conv2 = SAGEConv(hidden_features, hidden_features, aggr='mean')  # Second GraphSAGE layer
        self.fc = torch.nn.Linear(hidden_features, 1)  # Linear layer for output

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))  # Apply first GraphSAGE layer and ReLU activation
        x = F.relu(self.conv2(x, edge_index))  # Apply second GraphSAGE layer and ReLU activation
        x = global_mean_pool(x, batch)  # Global mean pooling
        x = self.fc(x)  # Apply final linear layer
        return x.squeeze()  # Ensure output is 1D
