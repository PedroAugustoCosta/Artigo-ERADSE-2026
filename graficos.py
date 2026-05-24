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
    de dados (6000) e as cores representam a distribuição das classes.
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
    # Note: removi a linha que transformava em porcentagem
    ax = df.plot(kind='bar', stacked=True, figsize=(12, 7), colormap="tab10", edgecolor="white")

    plt.title("Quantidade Total de Dados por Cliente e Distribuição de Classes", fontsize=14)
    plt.xlabel("Clientes", fontsize=12)
    plt.ylabel("Quantidade de Imagens (MNIST)", fontsize=12)
    plt.legend(title="Classes", loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Adicionar uma linha guia no 6000 para confirmar a consistência
    plt.axhline(y=6000, color='red', linestyle='--', alpha=0.3, label="Meta 6k")
    
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

