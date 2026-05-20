from collections import OrderedDict
from model import Net, test, train
import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from typing import Dict

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

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        optim = torch.optim.SGD(self.model.parameters(), lr=config['lr'], momentum=config['momentum'])
        train(self.model, self.trainloader, optim, config['local_epochs'], self.device)
        return self.get_parameters({}), len(self.trainloader.dataset), {}

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