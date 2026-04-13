from collections import OrderedDict
from model import Net, test, train
import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from typing import Dict, List, Tuple
#cria a classe de flower client com os seus atributos e funcionalidades
class FlowerCliente(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader,
                 num_classes)-> None:
        #atributos como os dados de treino, validação, o modelo de pytorch utilizado e o serviço utilizado
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader

        self.model = Net(num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        #extrai do pytorch o dicionario camada valor numerico associado a ela
        params_dict = zip(self.model.state_dict().keys(), parameters)
        #cria um um dicionario que pode ser utilizado pelo pytroch tranformando de numarray para torch.tensor
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        #carrega o modelo recebido
        self.model.load_state_dict(state_dict, strict=True)

    #pega os parametros do modelo recebido
    def get_parameters(self,config: Dict[str, Scalar]):

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):

        
        #copy parameters sent by the server into client's local model

        self.set_parameters(parameters)


        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']

        #escolhe o modelo de otimização

        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        #do a local training
        train(self.model, self.trainloader,optim,self.device)

        #pos treino pega os parametros e o tamanho da lista de dados de treino, para enviar ao servido a fim de que o defavg faça a média
        return self.get_parameters({}), len(self.trainloader), {}
    
    #avalia o modelo
    def evaluate(self,parameters: NDArrays, config: Dict[str,Scalar] ):
        #seta os parametros recebidos
        self.set_parameters(parameters)
        #testa o modelo
        loss, accuracy = test(self.model, self.valloader,self.device)
        #retorna o resultado
        return float(loss),len(self.valloader), {'accuracy':accuracy}
    

#cria o cliente ou os nós da rede federada

def generate_client_fn(trainloader,valloader,num_classes):
    #cria o cliente com os dados de treino, validação e o numero de classes
    def client_fn(cid: str):
        return FlowerCliente(trainloader=trainloader[int(cid)],
                             valloader=valloader[int(cid)],
                             num_classes=num_classes,).to_client()



    return client_fn