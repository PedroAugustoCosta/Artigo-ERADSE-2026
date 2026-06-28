import hydra
from omegaconf import DictConfig, OmegaConf
import flwr as fl

from client_dynamic import generate_client_fn
from dataset_non_iid import prepare_dataset
from server import get_on_fit_config,get_evaluate_fn

from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import pickle
from hydra.utils import instantiate
import logging


logging.getLogger("flwr").setLevel(logging.INFO)


logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("hydra").setLevel(logging.WARNING)
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    
    print(OmegaConf.to_yaml(cfg))

    
    trainloaders, validationloaders, testloader,_ = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )

   
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    
    strategy = instantiate(cfg.strategy)

    
    strategy.evaluate_fn = get_evaluate_fn(cfg.num_classes, testloader)
    strategy.on_fit_config_fn = get_on_fit_config(cfg.config_fit)

    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 1.0, 'num_gpus': 0.25}
    )

    
    save_path = HydraConfig.get().runtime.output_dir
    result_path = Path(save_path) / 'results.pkl'
    results = {'history': history}

    with open(str(result_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
   main()

