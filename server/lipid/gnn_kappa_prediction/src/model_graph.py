import os
from io import BytesIO
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import ast
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

from gatConv import GATPredictor
from gcnConv import GCNPredictor
from sageConv import SAGEPredictor


def safe_ast_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return s

# Function to encode string features as numerical values
def encode_features(features, feature_map):
    encoded_features = []
    for feature in features:
        encoded_feature = [feature_map.get(item, len(feature_map)) for item in feature]
        feature_map.update({item: i for i, item in enumerate(feature_map, start=len(feature_map)) if item not in feature_map})
        encoded_features.append(encoded_feature)
    return encoded_features


current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, "../data/Final_Dataset_for_Model_Train.csv")
data = pd.read_csv(file_path)

standard_feature_size = 1  # Set this based on your data analysis

def process_node_features(features):
    numeric_features = []
    for feature in features[1:]:  # Skip the first element which is node identifier
        if isinstance(feature, str):
            numeric_feature = feature_map.get(feature, len(feature_map))
            feature_map[feature] = numeric_feature
        else:
            numeric_feature = feature
        numeric_features.append(numeric_feature)

    # Pad the feature vector if it's shorter than the standard size
    if len(numeric_features) < standard_feature_size:
        numeric_features += [0] * (standard_feature_size - len(numeric_features))

    return numeric_features[:standard_feature_size]  # Ensure the feature vector is of standard size

feature_map = {}  # Dictionary to map strings to integers for node features
node_map = {}  # Dictionary to map node identifiers to integers for edges
unique_nodes = set()

other_columns = [
    'N Lipids/Layer',
    'N_water',
    'Temperature (K)',
    'Avg Membrane Thickness',
    'Kappa (BW-DCF)',
    'Kappa (RSF)']

data_list = []
max_feature_size = 0

for _, row in data.iterrows():
    # Convert string representations to Python objects
    node_features_list = safe_ast_literal_eval(row['Node Features'])
    edge_list = safe_ast_literal_eval(row['Edge List'])
    graph_features = safe_ast_literal_eval(row['Graph-Level Features'])

    for r in other_columns:
        graph_features.append(row[r])

    sample_feature_size = len(node_features_list[0]) - 1 + len(
        graph_features)  # Adjust based on your actual data structure

    if sample_feature_size > max_feature_size:
        max_feature_size = sample_feature_size

    # Create a mapping for all unique nodes in both node features and edge list
    for edge in edge_list:
        unique_nodes.update(edge)
    for features in node_features_list:
        unique_nodes.add(features[0])  # Assuming first element of each feature list is the node identifier

    node_map = {node_id: i for i, node_id in enumerate(unique_nodes)}

    # Prepare node features tensor
    # Initialize a tensor filled with zeros for each node
    num_nodes = len(unique_nodes)
    num_features = len(next(iter(node_features_list), []))  # Number of features per node
    node_features_tensor = torch.zeros((num_nodes, standard_feature_size), dtype=torch.float)

    for features in node_features_list:
        node_id = features[0]
        if node_id in node_map:
            node_idx = node_map[node_id]
            processed_features = process_node_features(features)
            node_features_tensor[node_idx] = torch.tensor(processed_features, dtype=torch.float)

    # Prepare edge index tensor
    edge_index_tensor = torch.tensor([[node_map[node] for node in edge] for edge in edge_list],
                                     dtype=torch.long).t().contiguous()

    graph_features_tensor = torch.tensor(graph_features, dtype=torch.float)

    y = torch.tensor([row['Kappa (q^-4)']], dtype=torch.float)
    # Expand and repeat graph-level features for each node
    expanded_graph_features = graph_features_tensor.unsqueeze(0).repeat(num_nodes, 1)

    # Concatenate graph-level features with node features
    result_tensor = torch.cat([node_features_tensor, expanded_graph_features], dim=1)

    current_size = result_tensor.size(-1)

    # print(current_size,max_feature_size)
    #
    # print(result_tensor.shape)

    if current_size < max_feature_size:
        padding_needed = max_feature_size - current_size
        pad_tensor = torch.full((result_tensor.size(0), padding_needed), 0, dtype=result_tensor.dtype)
        result_tensor = torch.cat([result_tensor, pad_tensor], dim=-1)

    # print(result_tensor.shape)

    data_list.append(Data(x=result_tensor, edge_index=edge_index_tensor, y=y))

# After creating feature_map and node_map during training
torch.save(feature_map, os.path.join(current_directory, '../models/feature_map_2.pth'))
torch.save(node_map, os.path.join(current_directory, '../models/node_map_2.pth'))
torch.save(unique_nodes, os.path.join(current_directory, '../models/unique_nodes_2.pth'))

# Split the data into training and validation sets
train_data = data_list[:int(0.8 * len(data_list))]
val_data = data_list[int(0.8 * len(data_list)):]

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

model1 = GATPredictor(input_features=9, hidden_features=32)
model2 = GCNPredictor(input_features=9, hidden_features=32)
model3 = SAGEPredictor(input_features=9, hidden_features=32)

import torch.optim as optim

def train_and_evaluate(model, train_loader, val_loader, epochs=1000):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        print(f"Epoch {epoch+1}, Train Loss: {total_train_loss / len(train_loader)}",end=" ")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                out = model(data)
                loss = criterion(out, data.y)
                total_val_loss += loss.item()
        print(f"Validation Loss: {total_val_loss / len(val_loader)}")


    # Evaluation
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for data in val_loader:
            out = model(data)
            predictions.extend(out.view(-1).tolist())
            actuals.extend(data.y.view(-1).tolist())

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    return mse, mae, r2

metrics = {}
for model, name in zip([model1, model2, model3], ['GAT', 'GCN', 'SAGE']):
    metrics[name] = train_and_evaluate(model, train_loader, val_loader)

# Plotting
metrics_df = pd.DataFrame(metrics, index=['MSE', 'MAE', 'R^2'])

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
metrics_df.loc['MSE'].plot(kind='bar', ax=ax[0], title='MSE Comparison')
metrics_df.loc['MAE'].plot(kind='bar', ax=ax[1], title='MAE Comparison')
metrics_df.loc['R^2'].plot(kind='bar', ax=ax[2], title='R^2 Comparison')
plt.show()

import matplotlib.pyplot as plt

# Assuming you have a function to get predictions for each model
def get_predictions(model, loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            predictions.extend(out.view(-1).tolist())
            actuals.extend(data.y.view(-1).tolist())
    return predictions, actuals

# Plot Actual vs. Predicted Values
import matplotlib.pyplot as plt
import torch

def plot_actual_vs_predicted_subplots(predictions_dict, actuals_dict, model_names):
    """
    Plots actual vs. predicted values for given models in subplots.

    Parameters:
    - predictions_dict: A dictionary containing model names as keys and their predictions as values.
    - actuals_dict: A dictionary containing model names as keys and their actual values as values.
    - model_names: A list of model names for which to plot the comparisons.
    """
    num_models = len(model_names)
    fig, axs = plt.subplots(1, num_models, figsize=(6 * num_models, 5))

    for i, model_name in enumerate(model_names):
        predictions = predictions_dict[model_name]
        actuals = actuals_dict[model_name]

        axs[i].scatter(actuals, predictions, alpha=0.5)
        axs[i].plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r')  # Diagonal line
        axs[i].set_title(f'Actual vs. Predicted for {model_name}')
        axs[i].set_xlabel('Actual Values')
        axs[i].set_ylabel('Predicted Values')

    plt.tight_layout()
    plt.show()



# Example usage for one model
predictions_gcn, actuals_gcn = get_predictions(model2, val_loader)
predictions_gat, actuals_gat = get_predictions(model1, val_loader)
predictions_sage, actuals_sage = get_predictions(model3, val_loader)


predictions_dict = {
    'GAT': predictions_gat,
    'GCN': predictions_gcn,
    'SAGE': predictions_sage
}

actuals_dict = {
    'GAT': actuals_gat,
    'GCN': actuals_gcn,
    'SAGE': actuals_sage
}

plot_actual_vs_predicted_subplots(predictions_dict,actuals_dict,['GAT', 'GCN', 'SAGE'])
