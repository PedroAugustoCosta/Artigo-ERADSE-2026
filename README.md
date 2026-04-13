# Estudo de Aprendizado Federado com Flower (flwr)

Este documento apresenta uma análise estruturada da biblioteca **Flower**, utilizando como base uma simulação de Aprendizado Federado (FL). A análise permeia o bloco principal (`main.py`) e suas interações com os demais módulos do sistema.

## 1. Imports e Dependências
O projeto utiliza bibliotecas específicas para organização de rede e gerenciamento de dicionários de estado:

* **Hydra**: Organiza a configuração da rede e subida dos experimentos.
* **OmegaConf**: Facilita a extração e manipulação de dicionários em moldes complexos (como o `state_dict` do modelo).
* **flwr (fl)**: Core da biblioteca Flower para Aprendizado Federado.
* **Módulos Locais**: Importação de funções para geração de clientes (`client`), preparação de dados (`dataset`) e lógica do servidor (`server`).

## 2. Configurações do Servidor (`base.yaml`)
As definições globais são gerenciadas pelo Hydra através de um arquivo YAML. Os principais parâmetros incluem:

* **num_rounds**: Quantidade de rodadas de treinamento global.
* **num_clients**: Total de clientes na rede (ex: 100).
* **batch_size**: Tamanho dos pacotes de dados para o treino.
* **num_clients_per_round_fit/eval**: Quantos clientes são selecionados para treinar e avaliar a cada rodada.
* **config_fit**: Hiperparâmetros como *Learning Rate* (lr) e *Momentum* passados aos clientes.

## 3. Preparação do Dataset
O dataset (MNIST no exemplo) é normalizado, embaralhado e dividido em partições para cada cliente:

```python
# Chamada principal no main.py
trainloaders, validationloaders, testloader = prepare_dataset(
    cfg.num_clients, 
    cfg.batch_size
)