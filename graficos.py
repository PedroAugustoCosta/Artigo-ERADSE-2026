import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import random

def plot_metadados_com_log(dados_da_distribuicao):
    """
    Plota o gráfico de metadados reais gerados no log 'dados_da_distribuicao'
    """
    num_clients = len(dados_da_distribuicao)
    clientes = list(range(num_clients))
    
    # Extrai os dados salvos no dicionário estruturado
    totais_imagens = [dados_da_distribuicao[c]["total_imagens"] for c in clientes]
    diversidade_classes = [dados_da_distribuicao[c]["qtd_classes"] for c in clientes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico 1: Volume Real de Imagens
    cores = ['salmon' if c < 5 else 'skyblue' for c in clientes]
    bars = ax1.bar(clientes, totais_imagens, color=cores, edgecolor='black', alpha=0.8)
    ax1.set_title("Volume Total de Imagens por Cliente (Real)", fontweight='bold')
    ax1.set_xlabel("ID do Cliente")
    ax1.set_ylabel("Quantidade de Dados")
    ax1.set_xticks(clientes)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 100, f"{yval}", ha='center', va='bottom', fontsize=9)

    # Gráfico 2: Diversidade Real de Classes Sorteada
    ax2.plot(clientes, diversidade_classes, marker='o', color='purple', linewidth=2, linestyle='--')
    ax2.set_title("Quantidade Real de Classes por Cliente", fontweight='bold')
    ax2.set_xlabel("ID do Cliente")
    ax2.set_ylabel("Número de Classes Diferentes")
    ax2.set_xticks(clientes)
    ax2.set_yticks(range(1, 11))
    ax2.grid(linestyle=':', alpha=0.6)
    
    plt.suptitle("Análise Estatística de Carga e Heterogeneidade dos Nós", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
   
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_absolute_distribution(dict_usuarios, trainset):
    """
    Gera um gráfico de barras empilhadas onde a altura é a quantidade total 
    de dados e as cores representam a distribuição das classes.
    Adiciona o valor total no topo de cada barra e o Total Global no gráfico.
    """
    num_clients = len(dict_usuarios)
    # Criamos a matriz de contagem: Clientes (linhas) x Classes (colunas)
    dist_matrix = np.zeros((num_clients, 10))
    
    labels = trainset.targets.numpy()

    for cliente_id, indices in dict_usuarios.items():
        for idx in indices:
            classe = labels[idx]
            dist_matrix[cliente_id][classe] += 1

    # Criar DataFrame para plotagem
    df = pd.DataFrame(dist_matrix, 
                      index=[f'Client {i}' for i in range(num_clients)],
                      columns=[f'Digit {i}' for i in range(10)])

    # Plotar gráfico de barras empilhadas (Stacked Bar Chart)
    ax = df.plot(kind='bar', stacked=True, figsize=(14, 8), colormap="tab10", edgecolor="white")

    plt.title("Quantidade Total de Dados por Cliente e Distribuição de Classes", fontsize=16, pad=20)
    plt.xlabel("Clientes", fontsize=14)
    plt.ylabel("Quantidade de Imagens (MNIST)", fontsize=14)
    
    # Legenda fora do gráfico à direita
    plt.legend(title="Classes", loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    # (Linha guia vermelha tracejada removida daqui)
  
    
    print("Gráfico salvo com sucesso na pasta 'resultados_graficos'!")
    # ---------------------------------------------------------
    # CÁLCULOS DOS TOTAIS
    # ---------------------------------------------------------
    totais_por_cliente = df.sum(axis=1)
    total_geral = totais_por_cliente.sum() # Soma de todos os clientes
    
    # Textos com os totais no topo das barras
    for i, total in enumerate(totais_por_cliente):
        ax.text(i, total + 100, f'{int(total)}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

    # ---------------------------------------------------------
    # CAIXA DE TEXTO COM O TOTAL GLOBAL (MOVIDA PARA A DIREITA)
    # ---------------------------------------------------------
    texto_total = f'Total Global de Dados: {int(total_geral)} amostras'
    
    # Estilo da caixinha (fundo amarelo claro, borda laranja)
    props_caixa = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange', alpha=0.9)
    
    # Posiciona a caixa no canto superior direito (x=0.98, y=0.96)
    # horizontalalignment='right' garante que a caixa cresça para a esquerda, não invadindo a legenda
    ax.text(0.98, 0.96, texto_total, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', horizontalalignment='right', 
            bbox=props_caixa, fontweight='bold', color='darkred')

    
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    os.makedirs("resultados_graficos", exist_ok=True)
    plt.savefig("resultados_graficos/distribuicao_non_iid.png", dpi=300, bbox_inches='tight')
    plt.savefig("resultados_graficos/distribuicao_non_iid.pdf", bbox_inches='tight') # Formato vetorial perfeito para Latex/Overleaf



import matplotlib.pyplot as plt
import json
import os

def plot_all_strategies(results_dir="."):
    # Lista de arquivos para carregar (ajuste os nomes conforme seus arquivos salvos)
    strategy_files = {
        "FedAvg": "resultados_FedAvg.json",
        "FedDynamic": "resultados_AlgoritmoFedDynamic.json",
        "FSVRG": "resultados_AlgoritmoFSVRG3.json",
        "CO-OP": "resultados_AlgoritmoCOOP.json",
        "FedAdp": "resultados_AlgoritmoFedADAP.json",
        "fedprox": "resultados_FedProx.json"
    }

    plt.figure(figsize=(10, 6))
    cores = {
    "FedAvg": "blue",
    "FedDynamic": "red",
    "FSVRG": "green",
    "CO-OP": "orange",
    "FedAdp": "purple",
    "fedprox": "gray"
}
    plt.style.use('seaborn-v0_8-paper') # Estilo limpo e profissional
    plt.rcParams.update({'font.size': 12}) # Fonte maior para leitura fácil
    for name, filename in strategy_files.items():
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
                # O eixo X é o número de rounds
                rounds = range(1, len(data["accuracy"]) + 1)
                plt.plot(rounds, data["accuracy"], marker='o', label=name, color=cores.get(name, 'black'))

    plt.title("Convergência: Acurácia por Round de Comunicação", fontsize=14)
    plt.xlabel("Communication Rounds", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparativo_algoritmos.png")
    plt.show()

plot_all_strategies()