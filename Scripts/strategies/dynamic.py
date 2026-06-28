import flwr as fl
from flwr.server.strategy import Strategy
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple, Union
import numpy as np
from flwr.server.strategy.aggregate import aggregate
from scipy.stats import wasserstein_distance
import json
class AlgoritmoFedDynamic(fl.server.strategy.FedAvg):
    def __init__(self, taxa_aprendizado_global: float , beta: float ,*args, **kwargs):
            
            super().__init__(*args,**kwargs)
            
            
            self.versao_global = 0 
            
            
            
            self.taxa_aprendizado_global = taxa_aprendizado_global
            self.current_global_weights = None
            self.beta = beta
            self.ultima_acuracia_global = 0.0
            self.ultimos_pesos_pk = None
            self.ultima_acuracia_global = 0.0
            self.acuracia_media_atual = 0.0
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """
        Injeta a versão global atual na configuração enviada aos clientes.
        """
        
        configuracoes = super().configure_fit(server_round, parameters, client_manager)
        
        
        # Verifica as condições do artigo para avisar os clientes
        is_first_round = bool(server_round == 1)
        diferenca_acc = float(abs(self.acuracia_media_atual - self.ultima_acuracia_global))
        recompute_df = bool(diferenca_acc > self.beta)
        
        for client_proxy, fit_ins in configuracoes:
            fit_ins.config["is_first_round"] = is_first_round
            fit_ins.config["recompute_df"] = recompute_df
            
        return configuracoes
            
        

    def aggregate_fit(self, server_round, results, failures):
            if not results:
                return None, {}
            if self.current_global_weights is None:
                self.current_global_weights = results[0][1].parameters
            global_model_atual = parameters_to_ndarrays(self.current_global_weights)
            
            parametros_cada_cliente=[]
            
         
                    
            
            listas_sk = []
            cdt_bruto=[]
            cdf_bruto=[]
            cpt_bruto=[]
            for i, (client, fit_res) in enumerate(results):
                parametros_cada_cliente.append(parameters_to_ndarrays(fit_res.parameters))
                cdt_bruto.append(fit_res.metrics.get('Dt', 0.0))
                cdf_bruto.append(fit_res.metrics.get('Df', 0.0))
                cpt_bruto.append(fit_res.metrics.get('accuracy', 0.0))
            self.ultima_acuracia_global = self.acuracia_media_atual
            self.acuracia_media_atual = np.mean(cpt_bruto)
                     
            cdt = normalize(cdt_bruto)
            cdf = normalize(cdf_bruto)
            cpt = normalize(cpt_bruto)          
                
            listas_sk = []
            for k in range(len(cdf)):
                # Usando a fórmula matemática corrigida e estável (multiplicativa)
                sk = cpt[k] * (1 - cdt[k]) * (1 - cdf[k])
                listas_sk.append(sk)
                
            somatorio_si = sum(listas_sk)
            listas_pk = [s / somatorio_si for s in listas_sk]

            # 4. Agrega os modelos usando o novo peso adaptativo (pk)
            w_global = [np.zeros_like(p) for p in parametros_cada_cliente[0]] 
            for i, modelo_cliente in enumerate(parametros_cada_cliente):
                for camada in range(len(w_global)):
                    w_global[camada] += listas_pk[i] * modelo_cliente[camada]
            
            delta_global = [w_new - w_old for w_new, w_old in zip(w_global, global_model_atual)]
            self.current_global_weights = ndarrays_to_parameters(w_global)
            
            return self.current_global_weights, {"delta_norm": float(np.linalg.norm(delta_global[0]))}
                
            


def calcula_data_quality(histograma,distribuicao_global):
      

      distancia_local = np.array(histograma)/np.sum(histograma)


      distancia = wasserstein_distance(distancia_local,distribuicao_global)

      return distancia


def normalize(lista):
    min_v, max_v = min(lista), max(lista)
    if max_v == min_v: return [0.5 for _ in lista]
    return [ ((v - min_v) / (max_v - min_v)) for v in lista]
