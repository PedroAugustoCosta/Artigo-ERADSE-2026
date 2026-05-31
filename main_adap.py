import hydra
from omegaconf import DictConfig, OmegaConf
import flwr as fl

from client_ADAP import generate_client_fn
from dataset_non_iid import prepare_dataset
from server import get_on_fit_config,get_evaluate_fn

from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import pickle
from hydra.utils import instantiate
import logging

# Mantenha o Flower no nível INFO para ver o resumo final
logging.getLogger("flwr").setLevel(logging.INFO)

# Silencie apenas o ruído do Ray e do Hydra (nível WARNING ou ERROR)
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("hydra").setLevel(logging.WARNING)
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # Print da configuração atual para registro
    print(OmegaConf.to_yaml(cfg))

    # Prepara os datasets
    trainloaders, validationloaders, testloader,_ = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )

    # Cria o client_fn com os dados preparados
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    # Instancia a estratégia definida no YAML
    strategy = instantiate(cfg.strategy)

    # Injeção das funções necessárias (que não foram instanciadas via YAML)
    strategy.evaluate_fn = get_evaluate_fn(cfg.num_classes, testloader)
    strategy.on_fit_config_fn = get_on_fit_config(cfg.config_fit)

    # Inicia a simulação
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 1.0, 'num_gpus': 0.25}
    )

    # Salva o histórico de resultados
    save_path = HydraConfig.get().runtime.output_dir
    result_path = Path(save_path) / 'results.pkl'
    results = {'history': history}

    with open(str(result_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
   main()

