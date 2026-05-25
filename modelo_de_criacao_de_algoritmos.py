import flwr as fl
from flwr.server.strategy import Strategy
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple, Union
import numpy as np





class SuaNovaEstrategia(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        
        # 1. PARTE UNIVERSAL: Extrair pesos e contagem de amostras
        pesos_para_agregar = []
        for client_proxy, fit_res in results:
            w_k = parameters_to_ndarrays(fit_res.parameters)
            D_k = fit_res.num_examples
            
            # 2. PARTE EXCLUSIVA: Sua lógica matemática aqui
            # (Pode ser o ângulo do gradiente, idade, perda local, etc.)
            peso_especifico = self.calcular_meu_peso_customizado(fit_res)
            
            # 3. PARTE UNIVERSAL: Preparar para a função aggregate
            pesos_para_agregar.append((w_k, D_k * peso_especifico))
            
        # 4. PARTE UNIVERSAL: Agregação final
        return aggregate(pesos_para_agregar), {}