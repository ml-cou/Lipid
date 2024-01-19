import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

current_directory = os.path.dirname(__file__)


# GCN Model Definition
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.fc1 = torch.nn.Linear(num_node_features, 16)
        self.fc2 = torch.nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# Function to load and preprocess data
def load_data():
    # Load the data
    file_path = os.path.join(current_directory, 'Lipid_Composition_Nodes_Only_final.csv')
    data = pd.read_csv(file_path)

    def convert_to_set(row):
        return set(row.strip('{}').replace("'", "").split(', ')) if row.strip('{}') else set()

    data['All Node Features'] = data['All Node Features'].apply(convert_to_set)

    # Encoding 'All Node Features' as the target
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(data['All Node Features'])

    # One-hot encoding for 'Lipid Comp'
    lipid_comp_encoder = OneHotEncoder(sparse=False)
    X_lipid_comp = lipid_comp_encoder.fit_transform(data[['Lipid Comp']])

    return X_lipid_comp, Y, mlb, lipid_comp_encoder

# Function to train the model
def train_model(X, Y, num_classes):
    X_tensor = torch.tensor(X, dtype=torch.float)
    Y_tensor = torch.tensor(Y, dtype=torch.float)

    model = GCN(num_node_features=X_tensor.shape[1], num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = F.binary_cross_entropy_with_logits(out, Y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return model

# Main function for GCN node prediction
def gcn_node_prediction(lipid_comp):
    X, Y, mlb, lipid_comp_encoder = load_data()
    num_classes = len(mlb.classes_)
    model = train_model(X, Y, num_classes)
    predicted_node_features = predict_node_features(model, lipid_comp_encoder, mlb, lipid_comp)
    # Save the model and encoders for later use
    torch.save(model.state_dict(),     os.path.join(current_directory, 'models\gcn_model.pth'))
    pd.to_pickle(lipid_comp_encoder,     os.path.join(current_directory, 'models\lipid_comp_encoder.pkl'))
    pd.to_pickle(mlb,     os.path.join(current_directory, 'models\mlb.pkl'))

    return list(predicted_node_features)

# Prediction function
def predict_node_features(model, lipid_comp_encoder, mlb, lipid_comp):
    model.eval()
    with torch.no_grad():
        input_data = pd.DataFrame({'Lipid Comp': [lipid_comp]})
        input_encoded = lipid_comp_encoder.transform(input_data)
        node_data_tensor = torch.tensor(input_encoded[0], dtype=torch.float).unsqueeze(0)
        output = model(node_data_tensor)
        predicted_features = torch.sigmoid(output).squeeze(0)
        predicted_features = (predicted_features > 0.5).int().numpy()
        predicted_features_array = np.array([predicted_features])
        predicted_feature_names = mlb.inverse_transform(predicted_features_array)[0]
        return set(predicted_feature_names)

