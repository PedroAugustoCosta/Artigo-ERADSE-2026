import hydra
from omegaconf import DictConfig, OmegaConf
import flwr as fl

from client import generate_client_fn
from dataset_non_iid import prepare_dataset
from server import get_on_fit_config,get_evaluate_fn

from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import pickle
from hydra.utils import instantiate
import logging
import glob
import os
# Mantenha o Flower no nível INFO para ver o resumo final
logging.getLogger("flwr").setLevel(logging.INFO)

# Silencie apenas o ruído do Ray e do Hydra (nível WARNING ou ERROR)
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("hydra").setLevel(logging.WARNING)
from client import generate_client_fn as gen_fedavg
from client_FSVRG import generate_client_fn as gen_fsvrg
from client_dynamic import generate_client_fn as gen_dynamic
from client_ADAP import generate_client_fn as gen_adap
from client_coop import generate_client_fn as gen_coop
from client import generate_client_fn as gen_fedprox
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # ... (preparo dos dados) ...
    arquivos_memoria = glob.glob("memoria_cliente_*.json") + glob.glob("snapshot_cliente_*.pkl")
    
    for arquivo in arquivos_memoria:
        try:
            os.remove(arquivo)
        except OSError as e:
            print(f"Erro ao deletar {arquivo}: {e}")
    print(f"Faxina concluída! {len(arquivos_memoria)} arquivos temporários deletados.")
    # 2. Descobre qual estratégia o Hydra carregou lendo a string do _target_
    alvo_estrategia = HydraConfig.get().runtime.choices.strategy.lower()
    trainloaders, validationloaders, testloader,_ = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )
    # 3. Escolhe o cliente correto com um if/else
    if "fsvrg" in alvo_estrategia:
        client_fn = gen_fsvrg(trainloaders, validationloaders, cfg.num_classes)
    elif "dynamic" in alvo_estrategia:
        client_fn = gen_dynamic(trainloaders, validationloaders, cfg.num_classes)
    elif "adap" in alvo_estrategia:
        client_fn = gen_adap(trainloaders, validationloaders, cfg.num_classes)
    elif "coop" in alvo_estrategia:
        client_fn = gen_coop(trainloaders, validationloaders, cfg.num_classes)
    elif "fedavg" in alvo_estrategia:
        client_fn = gen_fedavg(trainloaders, validationloaders, cfg.num_classes)
    else:
        client_fn=gen_fedprox(trainloaders,validationloaders,cfg.num_classes)
    print(OmegaConf.to_yaml(cfg))

   
    

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
    import json


    resultados = {
        "accuracy": [acc for round_idx, acc in history.metrics_centralized["accuracy"]],
        "loss": [loss for round_idx, loss in history.losses_centralized]
    }

    # Salva em um arquivo .json com o nome da estratégia
    with open(f"resultados_{cfg.strategy._target_.split('.')[-1]}.json", "w") as f:
        json.dump(resultados, f)

    # Salva o histórico de resultados
    save_path = HydraConfig.get().runtime.output_dir
    result_path = Path(save_path) / 'results.pkl'
    results = {'history': history}

    with open(str(result_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
   main()

