import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Parameters
from flwr.server.client_manager import ClientManager
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from flwr.server.strategy.aggregate import aggregate
# A estratégia deve herdar explicitamente, mas vamos garantir os métodos obrigatórios
class AlgoritmoFSVRG(FedAvg):
    def __init__(self, learning_rate=0.01, snapshot_frequency=10, **kwargs):
        fedavg_params = {
            'fraction_fit': kwargs.get('fraction_fit', 1.0),
            'fraction_evaluate': kwargs.get('fraction_evaluate', 1.0),
            'min_fit_clients': kwargs.get('min_fit_clients', 2),
            'min_evaluate_clients': kwargs.get('min_evaluate_clients', 2),
            'min_available_clients': kwargs.get('min_available_clients', 2),
        }
        super().__init__(**fedavg_params)
        
        # 3. Inicializa os seus parâmetros customizados
        self.learning_rate = learning_rate
        self.snapshot_frequency = snapshot_frequency
        self.global_grad = None
        self.w_t = None

    # Forçar a implementação do initialize_parameters para evitar a busca no Hydra
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None

    def num_fit_clients(self, num_available_clients):
        return super().num_fit_clients(num_available_clients)

    def configure_fit(self, server_round, parameters, client_manager):
        # 1. Define se é snapshot
        is_snapshot = (server_round % self.snapshot_frequency == 0)
        print(f"[SERVER] Enviando global_grad com len: {len(self.global_grad) if self.global_grad else 0}")
        
        # 2. Se for snapshot, congela o modelo atual (w_t)
        if is_snapshot:
            self.w_t = parameters_to_ndarrays(parameters)
            print(f"\n[FSVRG] Rodada {server_round}: Iniciando Snapshot.")
        
        # 3. Converte os pesos para preparar o envio
        pesos_atuais = parameters_to_ndarrays(parameters)
        
        # 4. Se tivermos gradiente global, concatena para enviar ao cliente (redução de variância)
        if self.global_grad is not None:
            params_com_gradiente = pesos_atuais + self.global_grad
            parametros_a_enviar = ndarrays_to_parameters(params_com_gradiente)
        else:
            parametros_a_enviar = parameters

        # 5. Configura o fit dos clientes
        configs = super().configure_fit(server_round, parametros_a_enviar, client_manager)
        
        for _, fit_ins in configs:
            fit_ins.config["is_snapshot"] = is_snapshot
            
        return configs

    def aggregate_fit(self, server_round, results, failures):
        
        if not results:
            return None, {}
        is_snapshot = results[0][1].metrics.get("is_snapshot", False)
        if is_snapshot:
         
            self.global_grad = self.calcular_media_gradientes(results)
            
            return ndarrays_to_parameters(self.w_t), {}
        else:
            return self.fsvrg_update(results)

    def calcular_media_gradientes(self, results):
        gradientes_para_agregar = []
        for _, fit_res in results:
            grad_local = parameters_to_ndarrays(fit_res.parameters)
            gradientes_para_agregar.append((grad_local, fit_res.num_examples))
        return aggregate(gradientes_para_agregar)

    def fsvrg_update(self, results):
        pesos_para_agregar = []
        for _, fit_res in results:
            w_k = parameters_to_ndarrays(fit_res.parameters)
            pesos_para_agregar.append((w_k, fit_res.num_examples))
            
        w_novo_global = aggregate(pesos_para_agregar)
        return ndarrays_to_parameters(w_novo_global), {}