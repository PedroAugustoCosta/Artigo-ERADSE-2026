from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor,Normalize,Compose
from torch.utils.data import random_split, DataLoader
import torch

def get_MNIST(data_path:str = './data'):
    tr = Compose([ToTensor() ,Normalize((0.1307,),(0.3081,))])
    trainset = MNIST(data_path,train=True,download=True,transform=tr)
    testset = MNIST(data_path,train=False,download=True,transform=tr)


    return trainset,testset



def prepare_dataset(num_partitions : int ,
                    batch_size: int,
                     val_ration : float = 0.1):

    trainset, testset=get_MNIST()

    #split trainset into 'num_partitions' trainsets

    num_images = len(trainset)// num_partitions
    partiton_len = (num_images) * num_partitions
    num_images_per_partition = len(trainset) // num_partitions
    partition_sizes = [num_images_per_partition] * num_partitions
    remainder = len(trainset) % num_partitions
    partition_sizes[-1] += remainder
    gen = torch.Generator().manual_seed(2023)
    trainsets = random_split(trainset, partition_sizes, generator=gen)
    trainloaders=[]
    valloaders=[]
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ration*num_total)
        num_train = num_total - num_val

        for_train, for_val =random_split(trainset_, [num_train,num_val], generator=gen)

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers = 2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers = 2))
    
    testloader= DataLoader(testset,batch_size=128)

    return  trainloaders, valloaders, testloader