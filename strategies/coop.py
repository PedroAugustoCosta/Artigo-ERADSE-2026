import flwr as fl
from flwr.server.strategy import Strategy
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple, Union
import numpy as np

class AlgoritmoCOOP(fl.server.strategy.FedAvg):
    def __init__(self, taxa_aprendizado_global: float = 1.0, **kwargs):
        
        self.versao_global = 0 
        self.taxa_aprendizado_global = taxa_aprendizado_global
        super().__init__(**kwargs)
        
        self.versao_global = 0 
        self.taxa_aprendizado_global = taxa_aprendizado_global

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """
        Injeta a versão global atual na configuração enviada aos clientes.
        """
        
        configuracoes = super().configure_fit(server_round, parameters, client_manager)
        
        
        for client_proxy, fit_ins in configuracoes:
            fit_ins.config["versao_global"] = self.versao_global
            
        return configuracoes

    def aggregate_fit(self, server_round: int, results, failures):
        if not results:
            return None, {}

        print(f"\n[CO-OP] Agregando Rodada {server_round} | Versão Global Atual: {self.versao_global}")

        pesos_para_agregar = []

      
        for client_proxy, fit_res in results:
            D_k = fit_res.num_examples
            
    
            versao_treinada = fit_res.metrics.get("versao_treinada", self.versao_global)
            
       
            atraso = self.versao_global - versao_treinada
            
        
            atraso = max(0, atraso) 

        
            fator_staleness = 1.0 / (1.0 + atraso)
            
            
            peso_final = D_k * fator_staleness
            
           
            print(f"Cliente {client_proxy.cid} | Atraso: {atraso} | Fator aplicado: {fator_staleness:.2f}")

            parametros_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            pesos_para_agregar.append((parametros_ndarrays, peso_final))

      
        from flwr.server.strategy.aggregate import aggregate
        agregacao_final = aggregate(pesos_para_agregar)

       
        self.versao_global += 1

        parameters_agregados = ndarrays_to_parameters(agregacao_final)
        return parameters_agregados, {}