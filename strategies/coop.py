import flwr as fl
from flwr.server.strategy import Strategy
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple, Union
import numpy as np

class AlgoritmoCOOP(fl.server.strategy.FedAvg):
    def __init__(self, taxa_aprendizado_global: float = 1.0, **kwargs):
        # 1. Inicializa os seus parâmetros específicos do CO-OP
        self.versao_global = 0 
        self.taxa_aprendizado_global = taxa_aprendizado_global
        super().__init__(**kwargs)
        # O Servidor guarda a "idade" do modelo global
        self.versao_global = 0 
        self.taxa_aprendizado_global = taxa_aprendizado_global

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """
        Injeta a versão global atual na configuração enviada aos clientes.
        """
        # Chama a configuração original para selecionar os clientes
        configuracoes = super().configure_fit(server_round, parameters, client_manager)
        
        # Adiciona a versão global no dicionário de config de cada cliente
        for client_proxy, fit_ins in configuracoes:
            fit_ins.config["versao_global"] = self.versao_global
            
        return configuracoes

    def aggregate_fit(self, server_round: int, results, failures):
        if not results:
            return None, {}

        print(f"\n[CO-OP] Agregando Rodada {server_round} | Versão Global Atual: {self.versao_global}")

        pesos_para_agregar = []

        # Para cada cliente que retornou resultados
        for client_proxy, fit_res in results:
            D_k = fit_res.num_examples
            
            # 1. Recupera qual versão do modelo o cliente usou
            versao_treinada = fit_res.metrics.get("versao_treinada", self.versao_global)
            
            # 2. Calcula o Atraso (Staleness / Aging)
            atraso = self.versao_global - versao_treinada
            
            # Se por algum motivo o atraso for negativo, zeramos
            atraso = max(0, atraso) 

            # 3. Calcula a função de atenuação (Penalidade por ser um cliente lento)
            # Fórmulas comuns da literatura: 1 / (1 + atraso) ou exp(-atraso)
            fator_staleness = 1.0 / (1.0 + atraso)
            
            # 4. Peso final do cliente (Quantidade de Dados * Penalidade de Atraso)
            peso_final = D_k * fator_staleness
            
            # Imprime para o seu relatório
            print(f"Cliente {client_proxy.cid} | Atraso: {atraso} | Fator aplicado: {fator_staleness:.2f}")

            parametros_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            pesos_para_agregar.append((parametros_ndarrays, peso_final))

        # Agrega os pesos usando a média ponderada com o fator de atraso
        from flwr.server.strategy.aggregate import aggregate
        agregacao_final = aggregate(pesos_para_agregar)

        # 5. Atualiza a versão global pois um novo modelo foi gerado
        self.versao_global += 1

        parameters_agregados = ndarrays_to_parameters(agregacao_final)
        return parameters_agregados, {}