import os

from django.http import JsonResponse
import pandas as pd
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Load the dataset
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, 'Lipid_Composition_Edges_Only_final.csv')
df = pd.read_csv(file_path)

def row_to_graph(row):
    edges = eval(row['All Edges'])
    unique_nodes = set(sum(edges, ()))
    node_mapping = {node: i for i, node in enumerate(unique_nodes)}
    edges_mapped = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in edges]
    edge_index = torch.tensor(edges_mapped, dtype=torch.long).t().contiguous()
    num_nodes = len(unique_nodes)
    x = torch.eye(num_nodes)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

graphs = [row_to_graph(row) for index, row in df.iterrows()]

class GNNModel(torch.nn.Module):
    def __init__(self, num_features):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

max_features = max([data.x.size(0) for data in graphs])
model = GNNModel(max_features)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    total_loss = 0
    for data in graphs:
        optimizer.zero_grad()
        num_padding = max_features - data.x.size(0)
        padded_features = F.pad(data.x, (0, num_padding), value=0)

        out = model(Data(x=padded_features, edge_index=data.edge_index, num_nodes=data.num_nodes))
        loss = criterion(out, padded_features)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss / len(graphs)}")


# Save the model
torch.save(model.state_dict(), os.path.join(current_directory, 'models\gnn_model.pth'))

# Function to load the model
def load_model(model_path, num_features):
    model = GNNModel(num_features)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to predict all edges for a given lipid composition
def predict_all_edges_for_lipid_comp(model, lipid_comp, node_labels, max_features):
    num_nodes = len(node_labels)
    edge_index = torch.combinations(torch.arange(num_nodes), 2).t()
    x = torch.eye(num_nodes)

    num_padding = max_features - x.size(0)
    padded_features = F.pad(x, (0, num_padding), value=0)

    data = Data(x=padded_features, edge_index=edge_index, num_nodes=num_nodes)

    with torch.no_grad():
        out = model(data)

    predicted_edge_scores = out
    threshold = 0.5
    predicted_edges = (predicted_edge_scores > threshold).nonzero()
    predicted_edges_list = [(node_labels[i], node_labels[j]) for i, j in predicted_edges]

    return predicted_edges_list

# Example Usage
loaded_model = load_model(os.path.join(current_directory, 'models\gnn_model.pth'), max_features)

from .gcn_molecule_node_only import gcn_node_prediction

def edge_pred(lipid_comp):


    def convert_to_json(data_list, name):
        node_dict = {}
        links_dict = {}

        for link in data_list:
            source, target = link

            if source not in node_dict:
                node_dict[source] = {"name": source, "value": 0, "children": [], "linkWith": []}
            if target not in node_dict:
                node_dict[target] = {"name": target, "value": 0, "children": [], "linkWith": []}

            node_dict[source]["linkWith"].append(target)
            node_dict[target]["linkWith"].append(source)

            links_dict[f"{source}_{target}"] = {"source": source, "target": target, "value": 1}

        root_name = name
        root_node = {"name": root_name, "value": 50, "children": list(node_dict.values()), "linkWith": []}

        for node in node_dict.values():
            root_node["linkWith"].extend(node["linkWith"])
            node["value"] = len(node["linkWith"])
            node["linkWith"] = [link["source"] for link in links_dict.values() if link["target"] == node["name"]]

        return root_node

    node_labels = gcn_node_prediction(lipid_comp)  # Replace with actual node labels
    predicted_edges = predict_all_edges_for_lipid_comp(loaded_model, lipid_comp, node_labels, max_features)

    predicted_edges = convert_to_json(predicted_edges, lipid_comp)

    if(predicted_edges):
        return JsonResponse (predicted_edges,safe=False)
    else:
        return JsonResponse ({},safe=False)
