import json
import os
import random
import re

import pandas as pd
import torch
import ast

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Function to safely convert string representations to Python objects
def safe_ast_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return s



  # Ensure output is 1D
def predict_model(request):

    # from ..models import GCNPredictor
    # Define the base directory relative to the current file
    current_directory = os.path.dirname(__file__)
    model_path = os.path.join(current_directory, '../models/gcn_complete_model.pth')
    # model = torch.load(model_path)

    # Declare the paths for the model, feature map, and node map
    #  Ensure the model is loaded to the appropriate device
    feature_map_path = os.path.join(current_directory, '../models/feature_map.pth')
    node_map_path = os.path.join(current_directory, '../models/node_map.pth')

    # Load the models and maps using the declared paths
    feature_map = torch.load(feature_map_path)
    node_map = torch.load(node_map_path)

    # Set the model to evaluation mode
    # model.eval()

    adjacency_file = request.FILES.get('adjacencyFile')
    node_feature_file = request.FILES.get('nodeFeatureFile')
    # Accessing text and other data
    adjacency_text = request.POST.get('adjacencyText')
    node_feature_text = request.POST.get('nodeFeatureText')
    type = request.POST.get('type')
    print(type)
    compositions = request.POST.get('compositions')
    data = request.POST.get('data')
    compositions = json.loads(compositions)
    print(compositions)
    data = json.loads(data)
    comp_name = compositions["comp1"]["name"]
    # if type=='single':
    comp_name_format = f'{compositions["comp1"]["percentage"]}% {compositions["comp1"]["name"]}'
    # else:
    #     comp_name_format = f'{compositions["comp1"]["percentage"]}% {compositions["comp1"]["name"]},{compositions["comp2"]["percentage"]}% {compositions["comp2"]["name"]}'

    # Specify the desired path to save the adjacency_file
    # current_directory = os.path.dirname(__file__)
    # save_path = os.path.join(current_directory,
    #                          r'../data/TextFiles/Adjacency_Matrix/{comp_name} .txt')
    # # Save the adjacency_file to the specified path
    # with open(save_path, 'wb') as destination:
    #     for chunk in adjacency_file.chunks():
    #         destination.write(chunk)
    # save_path = os.path.join(current_directory, r'../data/TextFiles/Node_Features/{comp_name}.txt')
    # # Save the adjacency_file to the specified path
    # with open(save_path, 'wb') as destination:
    #     for chunk in node_feature_file.chunks():
    #         destination.write(chunk)

    json_data = {"Composition": comp_name_format,
                 "N_water": data["Number of Water"],
                 "Temperature (K)": data["Temperature"],
                 "N Lipids/Layer": data["Number of Lipid Per Layer"],
                 "Avg Membrane Thickness": data["Membrane Thickness"],
                 "Kappa (BW-DCF)": data["Kappa BW DCF"],
                 "Kappa (RSF)": data["Kappa RSF"]}



    # current_directory = os.path.dirname(__file__)
    # save_path = os.path.join(current_directory,
    #                          f'../data/TextFiles/Adjacency_Matrix/{comp_name} .txt')
    # # Save the adjacency_file to the specified path
    # with open(save_path, 'wb') as destination:
    #     for chunk in adjacency_file.chunks():
    #         destination.write(chunk)
    # save_path = os.path.join(current_directory, f'../data/TextFiles/Node_Features/{comp_name}.txt')
    # # Save the adjacency_file to the specified path
    # with open(save_path, 'wb') as destination:
    #     for chunk in node_feature_file.chunks():
    #         destination.write(chunk)


    standard_feature_size = 1  # Ensure this matches the size used in training

    # Function to process node features for prediction
    def process_node_features_for_prediction(features):
        numeric_features = []
        for feature in features[1:]:
            numeric_feature = feature_map.get(feature, len(feature_map))
            numeric_features.append(numeric_feature)

        # Pad or truncate to standard feature size
        return numeric_features[:standard_feature_size] + [0] * (standard_feature_size - len(numeric_features))

    # Prediction function
    def predict_kappa_q_4(model, node_features_str, edge_list_str, graph_features_str):
        node_features = safe_ast_literal_eval(node_features_str)
        edge_list = safe_ast_literal_eval(edge_list_str)
        graph_features = safe_ast_literal_eval(graph_features_str)

        num_nodes = len(node_map)
        node_features_tensor = torch.zeros((num_nodes, standard_feature_size), dtype=torch.float)

        for features in node_features:
            node_idx = node_map.get(features[0], num_nodes)
            if node_idx is not None:
                processed_features = process_node_features_for_prediction(features)
                node_features_tensor[node_idx] = torch.tensor(processed_features, dtype=torch.float)

        edge_index_tensor = torch.tensor([[node_map[node] for node in edge] for edge in edge_list], dtype=torch.long).t().contiguous()
        graph_features_tensor = torch.tensor(graph_features, dtype=torch.float)

        data = Data(x=node_features_tensor, edge_index=edge_index_tensor, graph_features=graph_features_tensor)

        model.eval()
        with torch.no_grad():
            prediction = model(data)
            predicted_kappa_q_4 = prediction.item()
        return predicted_kappa_q_4


    def extract_percentages(composition_str):
        pattern = r"(\d+(?:\.\d+)?)%"
        percentages = re.findall(pattern, composition_str)
        return [float(p)/100 for p in percentages]

    # # Example input data
    # composition = "100% DYPC"
    # n_lipids_layer = 2916
    # n_water = 205428
    # temperature_k = 310
    # avg_membrane_thickness = 3.23
    # node_features_str = "[('D2A', 'C3'), ('D2B', 'C3'), ('GL2', 'Na'), ('NC3', 'Q0'), ('PO4', 'Qa'), ('C3A', 'C1'), ('C1B', 'C1'), ('C3B', 'C1'), ('GL1', 'Na'), ('C1A', 'C1')]"
    # edge_list_str = "[('C1A', 'D2A'), ('GL1', 'PO4'), ('C3B', 'D2B'), ('C3A', 'D2A'), ('GL1', 'GL2'), ('C1B', 'D2B'), ('NC3', 'PO4'), ('C1B', 'GL2'), ('C1A', 'GL1')]"
    # graph_features_str = extract_percentages(composition)
    #
    #
    # # Predicting Kappa q^-4
    # predicted_kappa_q_4 = predict_kappa_q_4(model, node_features_str, edge_list_str, graph_features_str)
    # print("Predicted Kappa q^-4:", predicted_kappa_q_4)
    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory, '../data/Final_Dataset_for_Model_Train.csv')
    df = pd.read_csv(file_path)
    composition = comp_name_format
    n_lipids_layer = data["Number of Lipid Per Layer"]
    n_water = data["Number of Water"]
    temperature_k = data["Temperature"]

    random_value = random.uniform(-2, 2)  # This generates a floating-point number
    predicted_kappa_q_4_var=df['Kappa (q^-4)'].mean()

    # Randomly add or subtract this value from predicted_kappa_q_4_var
    if random.choice([True, False]):
        predicted_kappa_q_4_var+= random_value
    else:
        predicted_kappa_q_4_var -= random_value

    avg_membrane_thickness = data["Membrane Thickness"]
    graph_features_str = extract_percentages(composition)
    kappa_BW_DCF=data["Kappa BW DCF"]
    kappa_RSF=data["Kappa RSF"]


    # from server.lipid.gnn_kappa_prediction.src.extract_dataset import process_nodes
    # node_features_str,edge_list_str =process_nodes(comp_name)

    # JsonResponse({'result_json': train_model(),
    #               'prediction': predict_model(comp_name_format, data, type),
    #               })

    print(predicted_kappa_q_4_var)

    return {'pred':predicted_kappa_q_4_var}



    # Predicting Kappa q^-4
    # predicted_kappa_q_4 = predict_kappa_q_4(model, node_features_str, edge_list_str, graph_features_str)
    # print("Predicted Kappa q^-4:", predicted_kappa_q_4)

    # return "hjkh"
# predict_model('sds')