import os
import json

import torch
import random
import numpy as np

import datasets.ss_transforms as sstr
from client import *
from server import *
from utils.args import get_parser
from datasets.idda import IDDADataset
from datasets.gtav import GTAVDataset
from models.deeplabv3 import deeplabv3_mobilenetv2
from utils.stream_metrics import StreamSegMetrics

#################################################################################################################

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_dataset_num_classes(dataset):
    if dataset == 'idda' or dataset == 'gtaV':
        return 16
    raise NotImplementedError

def model_init(args):
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    raise NotImplementedError

def get_transforms(args):

    if args.model == 'deeplabv3_mobilenetv2':

        if args.da == 'basic':
            train_transforms = sstr.Compose([
                sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
                sstr.ToTensor(),
                sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        elif args.da == 'advanced':
            train_transforms = sstr.Compose([
                sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
                sstr.RandomHorizontalFlip(),  #default P(flip)=0.5
                sstr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),     #as seen in FedDrive
                sstr.ToTensor(),
                sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        else:
            raise NotImplementedError
        
        test_transforms = sstr.Compose([
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
        raise NotImplementedError
    

    return train_transforms, test_transforms


def get_datasets(args):

    train_transforms, test_transforms = get_transforms(args)
    
    if args.dataset == 'idda':

        train_datasets = []
        root = 'data/idda'
        with open(os.path.join(root, 'train.json'), 'r') as f:
            all_data = json.load(f)

        for client_id in all_data.keys():
            train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms, client_name=client_id))

        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms, client_name='test_same_dom')
            
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms, client_name='test_diff_dom')

        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]

        return train_datasets, test_datasets
    
    elif args.dataset == 'gtaV':

        valid_datasets = []
        root = 'data/idda'
        with open("data/GTAV+Cityscapes/data/GTA5/train.txt",'r') as f:
            list_samples = [line.rstrip('\n') for line in f]
        train_dataset= GTAVDataset(root="data/GTAV+Cityscapes/data/GTA5",list_samples = list_samples,transform=train_transforms)
        
        with open(os.path.join(root, 'train.json'), 'r') as f:
            all_data = json.load(f)

        for client_id in all_data.keys():
            valid_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=test_transforms,
                                              client_name=client_id))
        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
        
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]

        return train_dataset,valid_datasets,test_datasets
    
    else:
        raise NotImplementedError

def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metric = StreamSegMetrics(num_classes, 'metric')
    else:
        raise NotImplementedError
    return metric

def gen_clients(args, train_datasets, test_datasets, model):
    clients = [[], []]
    for i, datasets in enumerate([train_datasets, test_datasets]):
        for ds in datasets:
            if args.mode == "SL":
                clients[i].append(Client(args, ds, model, test_client=(i==1)))         
            elif args.mode == "SSL":
                clients[i].append(ClientSSL(args, ds, model, test_client=(i==1)))
            else:
                raise NotImplementedError
    return clients[0], clients[1]    

def print_setup(args):
    print(" \n\n ############################################# \n\n I'm in the experiment:\n ",args)

def main(args_dict):

    args = get_parser(args_dict)
    print_setup(args)
    set_seed(args.seed)

    print(f'Initializing model...')
    model = model_init(args)
    model.cuda()
    print('Done.')

    print('Generate datasets...')
    train_datasets, test_datasets = get_datasets(args)
    print('Done.')

    metric = set_metrics(args)
    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)

    if args.mode == 'SL':
        server = Server(args, train_clients, test_clients, model, metric) 
    elif args.mode == 'SSL':
        model = torch.load(args.path_model_SSL)
        server = ServerSSL(args, train_clients, test_clients, teacher_model = model,student_model = model, metric = metric)
    
    server.train()

if __name__ == '__main__':
    main()