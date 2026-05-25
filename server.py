from omegaconf import DictConfig
from model import Net, test
import torch
from collections import OrderedDict


#função que ajusta o dicionário de configurações para cada rodada do servidor
def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        
        return {'lr':config.lr,'momentum':config.momentum,'local_epochs':config.local_epochs,"server_round": server_round}
    

    return fit_config_fn

def get_evaluate_fn(num_classes,testloader ):
    def evaluate_fn(server_rounds:int, parameters, config):
        #cria um modelo com a quantidade de classes passadas no arquivo de configuração
        model = Net(num_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        #usa a função torch.cuda_is_available para verificar se consegue rodar o modelo na gpu, se não roda na cpu
        

        #pareia os nomes das camadas com os valores numéricos 
        params_dict = zip(model.state_dict().keys(), parameters)
        #cria o dicionario de estado pegando as chaves e valores de params_dict, o flower manda como lista numpy e aqui convertemos de volta para tensor
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        #carrega o dicionario de estado de modelo restrito
        model.load_state_dict(state_dict, strict=True)
        #realiza os tests que retorna o valor de perda e acuracidade do modelo
        loss, accuracy = test(model,testloader,device)

        return loss, {'accuracy':accuracy}

    return evaluate_fn