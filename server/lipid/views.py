import json
import os
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
import matplotlib.pyplot as plt
from django.http import JsonResponse

from .gnn_kappa_prediction.src.model_graph import gen_comparison
from .gnn_kappa_prediction.src.predict_model_2 import predict_model as pm
from .gnn_kappa_prediction.src.train_model_2 import train_model
from .static.Predict_Value.Predict_value import predict_value
from .static.gnn_molecule_edge_only import edge_pred


plt.switch_backend('agg')


@csrf_exempt
# Create your views here.
@api_view(['GET','POST'])
def get_data(request):
    return JsonResponse(pm(request))

@api_view(['GET','POST'])
def get_model_comparison(request):
    graphs=gen_comparison()
    return JsonResponse(graphs)

@api_view(['GET','POST'])
def create_model(req):
    # return
    return train_model()
@api_view(['GET','POST'])
def evaluation(request):
    return JsonResponse(train_model())

@api_view(['GET','POST'])
def predict_model(req):
    data=json.loads(req.body)
    predict_model(data)
@api_view(['GET','POST'])
def prediction(req):
    data=json.loads(req.body)
    # Load and preprocess the dataset
    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory, 'final_dataset.csv')
    df = pd.read_csv(file_path)
    return predict_value(data,df)
@api_view(['GET','POST'])
def pred_edge(req):
    data = json.loads(req.body)
    print(data)
    return edge_pred(data.get("mol_name"))