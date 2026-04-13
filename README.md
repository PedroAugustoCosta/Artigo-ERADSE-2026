# [cite_start]Estudo de Aprendizado Federado com Flower (flwr) [cite: 1]

[cite_start]Este documento apresenta uma análise estruturada da biblioteca **Flower**, utilizando como base uma simulação de Aprendizado Federado (FL)[cite: 1, 3]. [cite_start]A análise permeia o bloco principal (`main.py`) e suas interações com os demais módulos do sistema[cite: 3].

## [cite_start]1. Imports e Dependências [cite: 4]
O projeto utiliza bibliotecas específicas para organização de rede e gerenciamento de dicionários de estado:

* [cite_start]**Hydra**: Organiza a configuração da rede e subida dos experimentos[cite: 11].
* [cite_start]**OmegaConf**: Facilita a extração e manipulação de dicionários em moldes complexos (como o `state_dict` do modelo)[cite: 12, 13].
* [cite_start]**flwr (fl)**: Core da biblioteca Flower para Aprendizado Federado[cite: 15].
* [cite_start]**Módulos Locais**: Importação de funções para geração de clientes (`client`) [cite: 14][cite_start], preparação de dados (`dataset`) [cite: 16] [cite_start]e lógica do servidor (`server`)[cite: 16].

## [cite_start]2. Configurações do Servidor (`base.yaml`) [cite: 17, 21]
[cite_start]As definições globais são gerenciadas pelo Hydra através de um arquivo YAML[cite: 18]. Os principais parâmetros incluem:

* [cite_start]**num_rounds**: Quantidade de rodadas de treinamento global[cite: 37, 47].
* **num_clients**: Total de clientes na rede (ex: 100)[cite: 38, 48].
* **batch_size**: Tamanho dos pacotes de dados para o treino[cite: 39, 49].
* **num_clients_per_round_fit/eval**: Quantos clientes são selecionados para treinar e avaliar a cada rodada[cite: 40, 41, 50, 51].
* [cite_start]**config_fit**: Hiperparâmetros como *Learning Rate* (lr) e *Momentum* passados aos clientes[cite: 43, 44, 45, 53].

## [cite_start]3. Preparação do Dataset [cite: 54]
[cite_start]O dataset (MNIST no exemplo) é normalizado, embaralhado e dividido em partições para cada cliente[cite: 52, 93]:

```python
# Chamada principal no main.py
trainloaders, validationloaders, testloader = prepare_dataset(
    cfg.num_clients, 
    cfg.batch_size
)


