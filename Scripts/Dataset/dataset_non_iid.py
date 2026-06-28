from torchvision.datasets import MNIST
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor,Normalize,Compose
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Subset
import torch
import numpy as np
import random
from flwr.serverapp.strategy import fedavg


def get_MNIST(data_path:str = './data'):
    tr = Compose([ToTensor() ,Normalize((0.1307,),(0.3081,))])
    trainset = MNIST(data_path,train=True,download=True,transform=tr)
    testset = MNIST(data_path,train=False,download=True,transform=tr)


    return trainset,testset

def distribuir_clientes_fixo(divisao_classes, num_clients=10, alpha=0.5):
    dict_usuarios = {i: [] for i in range(num_clients)}

    dados_da_distribuicao={}
    for cliente in range(num_clients):
        dados_da_distribuicao[cliente] =[]
       
        if cliente < 5:
            qtd_classes = random.randint(2, 5) 
        else:
            qtd_classes = random.randint(6, 10) 
        
        if qtd_classes > len(divisao_classes.keys()):
            qtd_classes =  len(divisao_classes.keys())

        classes_escolhidas = random.sample(list(divisao_classes.keys()), qtd_classes)
        
   
        proporcoes_classes = np.random.dirichlet([alpha] * qtd_classes)
        
        
        for  idx,cls in enumerate(classes_escolhidas):
            
          
            if cliente >5:
                qtd_da_classe = int(6000*proporcoes_classes[idx])
            else:
                qtd_da_classe= int(8000*proporcoes_classes[idx])

            indices=list(divisao_classes[cls])  
            dados_treino = indices[:qtd_da_classe]
            divisao_classes[cls] = divisao_classes[cls][qtd_da_classe:]
            if len(divisao_classes[cls]) ==0:
                del divisao_classes[cls]
            dict_usuarios[cliente].extend(dados_treino)  
        
        if cliente ==9 :
            for cls in divisao_classes.keys():
                dict_usuarios[cliente].extend(divisao_classes[cls])


        

        
        dados_da_distribuicao[cliente] = {
            "qtd_classes": qtd_classes,
            "classes_escolhidas": classes_escolhidas,
            "proporcoes_classes": proporcoes_classes,
            "total_imagens": len(dict_usuarios[cliente])
        }   

            
          
    
    return dict_usuarios,dados_da_distribuicao



def prepare_dataset(num_clients=10,batch_size=32):
    trainset, testset = get_MNIST()
    labels = trainset.targets.numpy()
    divisao_classes = {i: np.where(labels == i)[0] for i in range(10)}
    partes_testset = len(testset)//num_clients
    
    dict_test={i:[]for i in range(num_clients)}
    indices_test = list(range(len(testset)))
    np.random.shuffle(indices_test)

    for i in range(num_clients):

        start = i*partes_testset
        end = start + partes_testset
        dados_test = indices_test[start:end]
        
        
        dict_test[i]=dados_test


    dict_usuarios, dados_da_distribuicao = distribuir_clientes_fixo(divisao_classes)
    testloader= DataLoader(testset,batch_size=128)
    trainloaders, valloaders = prepare_dataloaders(trainset, testset, dict_usuarios, dict_test, batch_size=32)
 
    return trainloaders, valloaders ,testloader,dict_usuarios
    
   
import random
import numpy as np
import torch

def configurar_replicabilidade(seed=42):
    """
    Fixa as sementes de todas as bibliotecas para garantir que 
    os sorteios, distribuições e pesos da rede sejam idênticos em toda execução.
    """
    random.seed(seed)            # Trava o random.randint e random.sample
    np.random.seed(seed)         # Trava a Dirichlet do Numpy
    torch.manual_seed(seed)      # Trava a inicialização dos pesos da rede neural
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    print(f"Semente de replicabilidade fixada em: {seed}")

configurar_replicabilidade(42)

   
from torch.utils.data import DataLoader, Subset

def prepare_dataloaders(trainset, testset, dict_usuarios_treino, dict_test, batch_size=32):
    """
    Transforma os dicionários de índices em DataLoaders prontos para o Flower.
    """
    trainloaders = []
    testloaders = [] # Usado para validação local de cada cliente
    
    num_clients = len(dict_usuarios_treino)
    
    for cliente_id in range(num_clients):
        # 1. Resgatamos as listas de índices que você já calculou
        indices_treino = dict_usuarios_treino[cliente_id]
        indices_teste = dict_test[cliente_id]
        
        # 2. Criamos as fatias do dataset original (Subset)
        subset_treino = Subset(trainset, indices_treino)
        subset_teste = Subset(testset, indices_teste)
        
        # 3. Colocamos as fatias na "esteira rolante" (DataLoader)
        loader_treino = DataLoader(subset_treino, batch_size=batch_size, shuffle=True, num_workers=0)
        loader_teste = DataLoader(subset_teste, batch_size=batch_size, shuffle=False, num_workers=0)
        
        trainloaders.append(loader_treino)
        testloaders.append(loader_teste)
        
    return trainloaders, testloaders

trainset, testset = get_MNIST()
dict_usuarios_treino, dict_test,_,dict_usuarios= prepare_dataset(num_clients=10)

from graficos import plot_absolute_distribution

plot_absolute_distribution(dict_usuarios,trainset=trainset)