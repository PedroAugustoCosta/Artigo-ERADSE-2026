from collections import OrderedDict
from model import Net, test, train
import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from typing import Dict
import numpy as np
import json
from scipy.stats import wasserstein_distance
import os
def ler_memoria(cid):
    arquivo = f"memoria_cliente_{cid}.json"
    if os.path.exists(arquivo):
        with open(arquivo, 'r') as f:
            return json.load(f)
    return {'Dt': 0.0, 'Df': 0.0}

def salvar_memoria(cid, memoria_dict):
    arquivo = f"memoria_cliente_{cid}.json"
    with open(arquivo, 'w') as f:
        json.dump(memoria_dict, f)
class FlowerCliente(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes, cid):
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid 
        self.model = Net(num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.ultima_acuracia_global = 0.0
        self.ultimos_pesos_pk = None 
        self.beta = 0.02
        self.cached_Dt = 0.0
        self.cached_Df = 0.0
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
            peso_global_ultima_camada = parameters[-1]
            
            self.set_parameters(parameters=parameters)
            self.model.to(self.device)
            
            optim = torch.optim.SGD(self.model.parameters(), lr=config['lr'])       
            
            train(self.model, self.trainloader, optim, config['local_epochs'], self.device)
            loss, accuracy = test(self.model, self.valloader, self.device)
            
            is_first_round = config.get('is_first_round',False)
            mudanca_model= config.get('recompute_df',False)
            memoria = ler_memoria(self.cid)
            if is_first_round:
                histograma = get_distribuicao_percentual(self.trainloader, self.num_classes)
                distribuicao_ideal = np.ones(self.num_classes) / self.num_classes
                
                memoria['Dt'] = float(wasserstein_distance(histograma, distribuicao_ideal))
                
                pesos_treinados = self.get_parameters(config={})
                memoria['Df'] = float(np.linalg.norm(peso_global_ultima_camada - pesos_treinados[-1]))
            elif mudanca_model:
                pesos_treinados = self.get_parameters(config={})
                memoria['Df'] = float(np.linalg.norm(peso_global_ultima_camada - pesos_treinados[-1]))
                
            # Retorna recuperando os dados sãos e salvos da memória global
            return self.get_parameters(config={}), len(self.trainloader.dataset), {
                    'Dt': memoria['Dt'],
                    'Df': memoria['Df'],
                    'accuracy': float(accuracy)
                }
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        self.model.to(self.device)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader), {'accuracy': accuracy}



        
def generate_client_fn(trainloaders, valloaders, num_classes):
    def client_fn(cid: str) -> fl.client.Client:
        
        cliente_idx = int(cid)
        return FlowerCliente(
            trainloaders[cliente_idx], 
            valloaders[cliente_idx], 
            num_classes, 
            cid
        ).to_client()
    return client_fn

def get_distribuicao_percentual(trainloader, num_classes=10):
    contagens = np.zeros(num_classes, dtype=float)
    
   
    for _, labels in trainloader:
        for label in labels:
            contagens[int(label)] += 1
            
    
    total_amostras = np.sum(contagens)
    distribuicao = contagens / total_amostras
    
    return distribuicao