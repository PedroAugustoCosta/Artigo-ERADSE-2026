import flwr as fl
from flwr.server.strategy import Strategy
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple, Union
import numpy as np
from flwr.server.strategy.aggregate import aggregate
from scipy.stats import wasserstein_distance
import json
class AlgoritmoFedDynamic(fl.server.strategy.FedAvg):
    def __init__(self, taxa_aprendizado_global: float = 1.0, beta: float = 0.02, **kwargs):
            
            super().__init__(**kwargs)
            
            
            self.versao_global = 0 
            self.history_angles = {}
            self.alpha = 0.01
            
            
            self.taxa_aprendizado_global = taxa_aprendizado_global
            self.current_global_weights = None
            self.beta = beta
            self.ultima_acuracia_global = 0.0
            self.ultimos_pesos_pk = None
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """
        Injeta a versão global atual na configuração enviada aos clientes.
        """
        
        configuracoes = super().configure_fit(server_round, parameters, client_manager)
        
        
       
            
        return configuracoes

    def aggregate_fit(self, server_round, results, failures):
            if not results:
                return None, {}
            if self.current_global_weights is None:
                self.current_global_weights = results[0][1].parameters
            global_model_atual = parameters_to_ndarrays(self.current_global_weights)
            
            parametros_cada_cliente=[]
            
            histogramas_clientes = {
            i: json.loads(fit_res.metrics.get("histograma")) 
            for i, (client, fit_res) in enumerate(results)
        }
            
            hists = [np.array(h) / np.sum(h) for h in histogramas_clientes.values()]
                    
            dados_globais = np.mean(hists, axis=0)
            listas_sk = []
            cdt_bruto=[]
            cdf_bruto=[]
            cpt_bruto=[]
            for i ,(client,fit_res) in enumerate(results):
                  parametros_cada_cliente.append(parameters_to_ndarrays(fit_res.parameters))
                  histograma = histogramas_clientes[i]
                  ultima_camada = parametros_cada_cliente[i][-1]
                  
                  cdf_bruto.append(np.linalg.norm(global_model_atual[-1] - ultima_camada))
                  
                  
                  cdt_desnormalizado = calcula_data_quality(histograma,dados_globais)
                  cdt_bruto.append(cdt_desnormalizado)
                  
                  cpt_bruto.append(float(fit_res.metrics.get("accuracy", 0.0)))

            acuracia_media_atual = np.mean(cpt_bruto)
            diferenca_acc = abs(acuracia_media_atual - self.ultima_acuracia_global) 
              
            if diferenca_acc > self.beta or self.ultimos_pesos_pk is None:
                print(f"[FedDynamic] Round {server_round}: Recalculando pesos (Diff: {diferenca_acc:.4f} > {self.beta})")
                cdt=normalize(cdt_bruto)
                cdf=normalize(cdf_bruto)
                cpt=normalize(cpt_bruto)
                
                for k in range(len(cdf)):
                    sk = cpt[k] * (1 - cdt[k]) * (1 - cdf[k])
                    listas_sk.append(sk)
                somatorio_si= sum(listas_sk)
                
                self.ultimos_pesos_pk = [s / somatorio_si for s in listas_sk]
                self.ultima_acuracia_global = acuracia_media_atual
            
            else:
                print(f"[FedDynamic] Round {server_round}: Reusando pesos antigos (Diff: {diferenca_acc:.4f} <= {self.beta})")

            w_global=[np.zeros_like(p) for p in parametros_cada_cliente[0]] 
            listas_pk = self.ultimos_pesos_pk
            for i, modelo_cliente in enumerate(parametros_cada_cliente):
                for camada in range(len(w_global)):
                    w_global[camada] += listas_pk[i] * modelo_cliente[camada]
            
            
            delta_global = [w_new - w_old for w_new, w_old in zip(w_global, global_model_atual)]
            self.current_global_weights = ndarrays_to_parameters(w_global)
            return ndarrays_to_parameters(w_global), {"delta_norm": float(np.linalg.norm(delta_global[0]))}
            
            
            


def calcula_data_quality(histograma,distribuicao_global):
      

      distancia_local = np.array(histograma)/np.sum(histograma)


      distancia = wasserstein_distance(distancia_local,distribuicao_global)

      return distancia


def normalize(lista):
    min_v, max_v = min(lista), max(lista)
    if max_v == min_v: return [0.5 for _ in lista]
    return [0.1 + 0.9 * ((v - min_v) / (max_v - min_v)) for v in lista] #alterei isso do que estava no paper para ver se melhorava a performance
