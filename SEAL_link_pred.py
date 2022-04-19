# Reconnecting Top-l Relationships 
# Copyright (c) 2022 JadeRay. All Rights Reserved.
# Licensed under the Creative Commons BY-NC-ND 4.0 International License [see LICENSE for details]
# --------------------------------------------------------------------------------------------------
# Modified from SEAL_OGB (https://github.com/facebookresearch/SEAL_OGB)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------------------------------------------------

"""
The link prediction task with SEAL_OGB
"""
import os.path as osp

from tqdm import tqdm
import torch
from torch.nn import BCEWithLogitsLoss

from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from datasets import SEALDataset, SEALDatasetInMemory, toSEAL_pred_datalist
from models import DGCNN
from utils import read_temporary_graph_data


def train(loader):
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    y_pred, y_true = [], []
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))

    return roc_auc_score(torch.cat(y_true), torch.cat(y_pred))


@torch.no_grad()
def predict(loader, threshold=0.5):
    pred_edge_index = torch.zeros(0, 2).to(device)
    model.eval()
    
    for j, data in enumerate(loader):
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        logits = logits.view(-1).sigmoid()
        batch_mask = (logits > threshold).nonzero(as_tuple=True)[0]
        pred_edge_index = torch.cat((pred_edge_index, data.pred_edge[batch_mask]), dim=0)
    
    return pred_edge_index.type(torch.int).cpu()


T = [100, 80, 60, 40, 20]
datasets = ['MathOverflow', 'AskUbuntu', 'EmailEuCore', 'CollegeMsg']
BS = 8192
EPOCH = 50
LEARN_RATE = 0.0001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for dataset in datasets:
    for t in T:
        if osp.exists(f'data/SEALDataset/{dataset}/T{t}_pred_edge.pt'):
            print(f'data/SEALDataset/{dataset}/T{t}_pred_edge.pt is already exist.')
            continue
        print(f'\n\nLoading {dataset} Dataset with T={t}...')
        
        if dataset in ['WikiTalk', 'StackOverflow']: # Too big dataset may not process in Memory
            train_dataset = SEALDataset('data/SEALDataset', dataset, 'train', -1, 2, t)
            val_dataset = SEALDataset('data/SEALDataset', dataset, 'val', -1, 2, t)
            test_dataset = SEALDataset('data/SEALDataset', dataset, 'test', -1, 2, t)
            pred_dataset = [train_dataset.data, val_dataset.data, test_dataset.data]
            
            train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BS)
            test_loader = DataLoader(test_dataset, batch_size=BS)
        else:
            train_dataset = SEALDatasetInMemory('data/SEALDataset', dataset, 'train', 2, t)
            val_dataset = SEALDatasetInMemory('data/SEALDataset', dataset, 'val', 2, t)
            test_dataset = SEALDatasetInMemory('data/SEALDataset', dataset, 'test', 2, t)
            
            train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BS)
            test_loader = DataLoader(test_dataset, batch_size=BS)
        
        model = DGCNN(hidden_channels=32, num_layers=3, train_dataset=train_dataset).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARN_RATE)
        criterion = BCEWithLogitsLoss()
        
        if osp.exists(f'data/models/SEAL_{dataset}_T{t}.pth'):
            model.load_state_dict(torch.load(f'data/models/SEAL_{dataset}_T{t}.pth'))
        else:
            print(f'\nTraining {dataset} Dataset with T={t}...')
            best_val_auc = test_auc = 0
            for epoch in tqdm(range(1, EPOCH + 1)):
                loss = train(train_loader)
                val_auc = test(val_loader)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    test_auc = test(test_loader)
            print(f'Loss: {loss:.4f}, Val_best: {best_val_auc:.4f}, Test: {test_auc:.4f}')
                
            torch.save(model.state_dict(), f'data/models/SEAL_{dataset}_T{t}.pth')
        
        print(f'\nPredicting {dataset} Dataset with T={t}...')
        pred_graph = read_temporary_graph_data(test_dataset.graph_data_file_name, test_dataset.raw_file_timespan, t)[-1]
        assert pred_graph.number_of_nodes() > 30
        pred_data_list = toSEAL_pred_datalist(pred_graph, test_dataset.num_features)
        pred_loader = DataLoader(pred_data_list, batch_size=BS)
        pred_edge_index = predict(pred_loader, threshold=0.5)  
        torch.save(pred_edge_index, f'data/SEALDataset/{dataset}/T{t}_pred_edge.pt')
        
        print(f'\nCompleted {dataset} Dataset with T={t} operation.')
