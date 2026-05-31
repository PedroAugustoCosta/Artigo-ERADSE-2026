import flwr as fl
from flwr.server.strategy import Strategy
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple, Union
import numpy as np
from flwr.server.strategy.aggregate import aggregate
class AlgoritmoFedADAP(fl.server.strategy.FedAvg):
    def __init__(self, taxa_aprendizado_global: float = 1.0, **kwargs):
            # 1. Passa os argumentos padrão para a classe pai (FedAvg)
            super().__init__(**kwargs)
            
            # 2. Inicializa os parâmetros específicos do seu algoritmo (FedAdp)
            self.versao_global = 0 
            self.history_angles = {}
            self.alpha = 0.01
            
            # 3. Armazena a taxa de aprendizado global uma única vez
            self.taxa_aprendizado_global = taxa_aprendizado_global
            self.current_global_weights = None
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """
        Injeta a versão global atual na configuração enviada aos clientes.
        """
        # Chama a configuração original para selecionar os clientes
        configuracoes = super().configure_fit(server_round, parameters, client_manager)
        
        # Adiciona a versão global no dicionário de config de cada cliente
       
            
        return configuracoes

    def aggregate_fit(self, server_round, results, failures):
        
            if not results:
                return None, {}
            
            # 1. Obter o shape do modelo global atual
            
            
            amostras_dos_clientes = {}
            lista_de_deltas = []
            pesos = []
            total_amostras = sum([fit_res.num_examples for _, fit_res in results])

            # 2. LOOP 1: Coleta os modelos treinados e calcula os Deltas no servidor
            for i, (client, fit_res) in enumerate(results):
                lista_completa = parameters_to_ndarrays(fit_res.parameters)               
                mid = len(lista_completa) // 2               
                w_local = lista_completa[:mid]
                
                
                pesos.append(w_local)
                amostras_dos_clientes[i] = fit_res.num_examples
                
                if self.current_global_weights is None:
                    # Na rodada 1, ignoramos o cálculo do delta e usamos uma aproximação neutra
                    delta_i = [np.zeros_like(camada) for camada in w_local]
                else:
                    # A MÁGICA REAL: A partir da rodada 2, calcula a diferença
                    delta_i = lista_completa[mid:]
                    
                lista_de_deltas.append(delta_i)
                

            # 3. O resto do seu código FedAdp continua EXATAMENTE IGUAL a partir daqui!
            gradiente_global, gradientes_por_clientes = calcula_gradiente_global(
                lista_de_deltas, amostras_dos_clientes, total_amostras, lista_de_deltas[0], self.taxa_aprendizado_global
            )

       
            
            # 4. LOOP 2: Calcular similaridade e score (f_i)
            lista_f_i = {}
            for i, (client, fit_res) in enumerate(results):
                gradiente_global_flatten = np.concatenate([g.flatten() for g in gradiente_global])
                gradiente_cliente_flatten = np.concatenate([g.flatten() for g in gradientes_por_clientes[i]])
                
                # Calcular cosseno
                numerador = np.dot(gradiente_global_flatten, gradiente_cliente_flatten)
                denominador = np.linalg.norm(gradiente_global_flatten) * np.linalg.norm(gradiente_cliente_flatten) + 1e-8
                angulo = np.arccos(numerador / denominador)
            
                # Atualizar histórico
                if server_round == 1:
                    self.history_angles[i] = angulo
                else:
                    self.history_angles[i] = calcula_smoothed_angle(self.history_angles[i], server_round, angulo)
            
                # Calcular e guardar o f_i do cliente i
                f_i = calcula_f_i(self.history_angles[i], self.alpha)
                lista_f_i[i] = f_i
            
            # IMPORTANTE: Esta linha agora está FORA do loop! 
            # Ela normaliza todos os f_i coletados.
            weighting = calcular_ponderacao(lista_f_i,amostras_dos_clientes)
                
            # 5. LOOP 3: Agregação ponderada final
            modelo_global_final = [np.zeros_like(camada) for camada in pesos[0]]

            # CORREÇÃO: Nome da variável alterado para não sobrescrever a lista 'pesos'
            for i, modelo_cliente in enumerate(pesos):
                for camada in range(len(modelo_global_final)):
                    modelo_global_final[camada] += weighting[i] * modelo_cliente[camada]

            # 6. Retornar os novos pesos para o Flower
            self.current_global_weights = modelo_global_final
            return ndarrays_to_parameters(modelo_global_final), {}
        
       
def calcula_gradiente_global(lista_de_deltas, amostras_dos_clientes, total_amostras, deltas_primeiro_cliente, taxa_de_aprendizado_global):
    gradiente_global = [np.zeros_like(d) for d in deltas_primeiro_cliente]
    gradientes_por_cliente = {}
    
    for i, delta_i in enumerate(lista_de_deltas):
        peso = amostras_dos_clientes[i] / total_amostras
        
        # Divide cada matriz da camada pela taxa de aprendizado individualmente
        grad_local = [-camada / taxa_de_aprendizado_global for camada in delta_i]

        gradientes_por_cliente[i] = grad_local
        
        for camada in range(len(gradiente_global)):
            gradiente_global[camada] += peso * grad_local[camada]
            
    return gradiente_global, gradientes_por_cliente

def calcula_smoothed_angle(angulo_antigo, server_round, angulo_novo):
        return ((server_round - 1) / server_round) * angulo_antigo + (1 / server_round) * angulo_novo
        
def calcula_f_i(angulo,alpha):
        return  alpha * (1 - np.exp(-np.exp(-alpha * (angulo - 1))))


def calcular_ponderacao(f_i_values,amostras_dos_clientes):
    """
    f_i_values: dicionário ou lista com os scores calculados para cada cliente.

    """
    scores = {}
    for i in range(len(f_i_values)):
        f_theta = np.exp(f_i_values[i])
        scores[i] = f_theta* amostras_dos_clientes[i]

    total = sum(scores.values())
        
    # Normaliza: o peso de cada cliente é sua parcela no total de scores
    weighting = {k: v / total for k, v in scores.items()}
    return weighting