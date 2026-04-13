import hydra
from omegaconf import DictConfig, OmegaConf
from client import generate_client_fn
import flwr as fl
from dataset import prepare_dataset
from server import get_on_fit_config,get_evaluate_fn

#define o caminho e o nome da base de configurações
@hydra.main(config_path ="conf",config_name="base",version_base=None)
def main(cfg:DictConfig):
    #parse configs e get experiment output dir

    print(OmegaConf.to_yaml(cfg))

    #prepare your dataset
    #prepara o dataset e devolve os dados de treino, validação e teste
    trainloaders, validationloaders, testloader =prepare_dataset(cfg.num_clients,
    
    
                                                            cfg.batch_size)
    

    print(len(trainloaders),len(trainloaders[0].dataset))

    #define your clients
    #define e configura os clientes mandando os seus dados e as classes
    client_fn = generate_client_fn(trainloaders, validationloaders,cfg.num_classes)
    #define your strategy
    #define a estratégia do servidor como o treino irá ocorrer a cada rodada
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.00001,
                                         min_fit_clients=cfg.num_clients_per_round_fit,#clientes mexidos por rodada
                                         fraction_evaluate = 0.00001,
                                          min_evaluate_clients = cfg.num_clients_per_round_eval, #clientes avaliados por rodada
                                         min_available_clients=cfg.num_clients,
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),#pega os parametros de ajuste dos clientes
                                         evaluate_fn=get_evaluate_fn(cfg.num_classes,#avalia o modelo global
                                                                     testloader))
    

    #start sumulation

    history = fl.simulation.start_simulation(client_fn=client_fn,#inicia o cliente
                                             num_clients=cfg.num_clients,#inicia N clientes
                                             config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),#inicia o servidor por um num rounds

                                             strategy=strategy)#modela a estrategia do servidor
if __name__=="__main__":
    main()