# Estudo de Aprendizado Federado com Flower (flwr)

## 0 - Introdução
Para iniciar a análise e o estudo da biblioteca Flower, utilizei como base a simulação do tutorial oficial de 2023: *FL Simulation with Flower*. Este documento guia a análise do `flwr` (nome usado para importar a biblioteca) focando no bloco `main.py` e sua integração com os outros módulos.

## 1 - Imports
O projeto organiza as dependências da seguinte forma:

* **hydra**: Biblioteca para organizar a configuração da rede e subida dos experimentos.
* **omegaconf**: Usada para elaboração e extração de dicionários de configuração (como o dicionário de estado do modelo).
* **client**: Importa a função `generate_client_fn`.
* **flwr (fl)**: Core da biblioteca Flower.
* **dataset**: Bloco onde ocorre a partição e preparação dos dados.
* **server**: Importa as funções `get_on_fit_config` e `get_evaluate_fn`.

## 2 - Definição das Configurações do Servidor
As definições globais são gerenciadas pelo Hydra através do arquivo `base.yaml`:

* **num_rounds**: Número de rodadas de treinamento.
* **num_clients**: Total de clientes na simulação (100).
* **batch_size**: Tamanho dos pacotes de dados.
* **num_clients_per_round_fit**: Clientes selecionados para treino por round.
* **num_clients_per_round_eval**: Clientes selecionados para avaliação por round.
* **num_classes**: Número de categorias (10 para o MNIST).
* **config_fit**: Hiperparâmetros como `lr` (learning rate) e `momentum`.

## 3 - Preparação do Dataset
A função `prepare_dataset` no arquivo `dataset.py` é responsável por normalizar, embaralhar e dividir o dataset MNIST em pacotes para os 100 clientes, garantindo que cada nó tenha sua própria partição de treino e validação.

## 4 - Criação dos Clientes
A função `generate_client_fn` instancia objetos da classe `FlowerCliente`. Cada cliente possui:
* Dados de treino e validação locais.
* O modelo definido em `model.py` (`Net`).
* O dispositivo de hardware (`cpu` ou `cuda`).

### Métodos Principais do Cliente:
* **set_parameters**: Converte os pesos de NumPy (servidor) para Tensores (PyTorch).
* **get_parameters**: Converte os pesos do modelo local para NumPy para enviar ao servidor.
* **fit**: Realiza o treino local usando o otimizador **Stochastic Gradient Descent (SGD)**.
* **evaluate**: Testa o modelo global nos dados locais do cliente.

## 5 - Estratégia da Rede (FedAvg)
A estratégia `FedAvg` (Federated Averaging) define como o servidor agrega os conhecimentos. 
* **on_fit_config_fn**: Envia as configurações de ajuste para os clientes a cada rodada.
* **evaluate_fn**: Realiza a **avaliação global** no servidor usando o modelo agregado e o dataset de teste completo.

## 6 - Iniciação da Simulação
A função `start_simulation` coordena todo o processo, enviando as configurações e armazenando o histórico de perda e acurácia de cada rodada na variável `history`.