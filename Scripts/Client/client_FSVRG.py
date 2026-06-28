from collections import OrderedDict
from model import Net, test, train
import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from typing import Dict
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

import numpy as np
import pickle
import os
def salvar_snapshot(cid, gradiente):
    # Salva as matrizes do gradiente em um arquivo físico
    with open(f"snapshot_cliente_{cid}.pkl", "wb") as f:
        pickle.dump(gradiente, f)

def carregar_snapshot(cid):
    # Lê as matrizes do gradiente do arquivo físico
    arquivo = f"snapshot_cliente_{cid}.pkl"
    if os.path.exists(arquivo):
        with open(arquivo, "rb") as f:
            return pickle.load(f)
    return None
class FlowerCliente(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes, cid):
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid # Corrigido aqui
        self.model = Net(num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        print("Pesos carregados com sucesso!")
    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def calcular_gradiente_total(self):
        """
        Calcula o gradiente do modelo atual em relação a todo o dataset local.
        """
        # Certifique-se de que o modelo está no modo de avaliação ou treino (conforme necessário)
        self.model.eval()
        self.model.to(self.device)
        
        # Zera os gradientes acumulados
        self.model.zero_grad()
        
        # Acumula o gradiente sobre todo o dataset
        total_loss = 0.0
        for images, labels in self.trainloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            
            # Backward pass para calcular o gradiente desta amostra
            loss.backward()
            total_loss += loss.item()

        # Extrai os gradientes e transforma em uma lista de arrays numpy (para enviar ao servidor)
        num_batches = len(self.trainloader)
        gradientes = [(param.grad / num_batches).detach().cpu().numpy().copy() for param in self.model.parameters()]
        # Zera os gradientes novamente para não interferir no treino normal
        self.model.zero_grad()
        
        return gradientes
    
    def fit(self, parameters, config):
        from flwr.common import parameters_to_ndarrays
        
        
        pesos_referencia = self.get_parameters(config={})
        num_pesos = len(pesos_referencia)
        
        
        params_longos = parameters if isinstance(parameters, list) else parameters_to_ndarrays(parameters)
        
        
        pesos_modelo = params_longos[:num_pesos]
        grad_global = params_longos[num_pesos:] 
        
        
        self.set_parameters(pesos_modelo)
        
        grad_snapshot_local = carregar_snapshot(self.cid)
        print(f"[CLIENT {self.cid}] Norma pesos: {np.linalg.norm(pesos_modelo[0]) if len(pesos_modelo) > 0 else 0}")
        
        is_snapshot = config.get("is_snapshot", False)
        
        if is_snapshot:
           
            grad_snapshot = self.calcular_gradiente_total()
            
            
            salvar_snapshot(self.cid, grad_snapshot)
            
       
            return grad_snapshot, len(self.trainloader.dataset), {"is_snapshot": True}
        
        else:
            
            optim = torch.optim.SGD(self.model.parameters(), lr=config['lr'])
            
            
            if len(grad_global) == 0:
                grad_global = [np.zeros_like(p) for p in pesos_modelo]
                grad_snapshot_local = carregar_snapshot(self.cid)
            if grad_snapshot_local is not None:
                train_fsvrg(self.model, self.trainloader, optim, config['local_epochs'], 
                            self.device, grad_global, grad_snapshot_local)
            else:
                train(self.model, self.trainloader, optim, config['local_epochs'], self.device)
            
            return self.get_parameters(config={}), len(self.trainloader.dataset), {"is_snapshot": False}
    
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

def train_fsvrg(model, trainloader, optim, epochs, device, grad_global, grad_snapshot_local):
    model.train()
    model.to(device)
    
    
    if not grad_global: 
        grad_global = [np.zeros_like(p.detach().cpu().numpy()) for p in model.parameters()]
        
    g_snap = [torch.tensor(g).to(device) for g in grad_snapshot_local]
    g_glob = [torch.tensor(g).to(device) for g in grad_global]

    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optim.zero_grad()
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            
           
            norma_antes = torch.norm(list(model.parameters())[0].grad)
            
            
            for param, snap, glob in zip(model.parameters(), g_snap, g_glob):
                if param.grad is not None:
                    param.grad = param.grad -snap +glob
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optim.step()
            
            norma_depois = torch.norm(list(model.parameters())[0].grad)
            print(f"Norma antes/depois: {norma_antes:.4f} / {norma_depois:.4f}")