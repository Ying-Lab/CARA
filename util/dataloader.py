import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
class RNA(Dataset):#提供 coembeddding orembedding 标签 不确定性
    def __init__(self, data,targets,omic,barcodes,batch=None, indexs=None, dataoe=None,temperature=None, temp_uncr=None,transform=None, train=True,
                  target_transform=None, batch_num = None, no_class = None):
        super().__init__()
        
        self.targets = np.array(targets)
        # no_seen = self.targets.max()
        self.barcodes = barcodes
        self.data = torch.from_numpy(data).float()
        self.targets=torch.zeros(self.targets.shape[0], no_class).scatter_(1, torch.tensor(self.targets).view(-1, 1).long(), 1)
        # self.data = torch.log(self.data + 1.)   
        self.transform = transform
        self.target_transform = target_transform
        self.omic = np.array(omic) 
        self.omic = torch.zeros(self.omic.shape[0], 2).scatter_(1, torch.tensor(self.omic).view(-1,1).long(), 1)
        self.batch = np.array(batch)  
        self.batch = torch.zeros(self.batch.shape[0], batch_num).scatter_(1, torch.tensor(self.batch).view(-1,1).long(), 1)
        self.dataoe = dataoe  
        if temperature is not None:
            self.temp = temperature*np.ones(len(self.targets))
        else:
            self.temp = np.ones(len(self.targets))


        if temp_uncr is not None:
            self.temp[temp_uncr['index']] = temp_uncr['uncr']

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            if self.transform is not None:
                self.dataoe = self.dataoe[indexs]
            # print("self.data",self.data)
            self.targets = self.targets[indexs]
            # print("self.targets",self.targets)
            self.temp = self.temp[indexs]
            self.omic = self.omic[indexs]    
            self.barcodes = self.barcodes[indexs] 
            if batch is not None:
                self.batch = self.batch[indexs]  
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))
    def __len__(self):
            return len(self.data)
    def __getitem__(self, index):
        if self.transform is not None:
            emb = self.dataoe[index]
        else:
            emb = self.data[index]
        barcode = self.barcodes[index] 
        target = self.targets[index]
        omic = self.omic[index]   
        if self.batch is not None:
            batch = self.batch[index]   
            return emb, target, self.indexs[index], self.temp[index],omic,barcode,batch #omic不参与训练只用于验证时判断对atac细胞的准确率
        return emb, target, self.indexs[index], self.temp[index],omic,barcode


class RNA_TEST(Dataset):
    def __init__(self, data,targets,omic,barcodes,batch=None,dataoe=None,train=False,
                 transform=None, target_transform=None,
                 labeled_set=None,batch_num = None, no_class = None):
        super().__init__()
        self.targets = np.array(targets)
        # print(self.targets)        
        self.data = torch.from_numpy(data).float()
        self.targets = torch.zeros(self.targets.shape[0], no_class).scatter_(1, torch.tensor(self.targets).view(-1,1).long(), 1) 
        self.barcodes = barcodes
        # self.data = torch.log(self.data + 1.)
        self.transform = transform
        self.target_transform = target_transform
        self.omic = np.array(omic) 
        self.omic = torch.zeros(self.omic.shape[0], 2).scatter_(1, torch.tensor(self.omic).view(-1,1).long(), 1)
        if batch is not None:
            self.batch = np.array(batch)  
            self.batch = torch.zeros(self.batch.shape[0], batch_num).scatter_(1, torch.tensor(self.batch).view(-1,1).long(), 1)

        self.dataoe = dataoe  
        # print(len(labeled_set))
        indexs = []
        if labeled_set is not None:
            self.labeled_set = labeled_set
            # print(labeled_set)
            # self.labeled_set=map_names_to_indices(labeled_set,category_to_index)
            # print(self.labeled_set)
            for i in range(no_class):#
                idx = np.where(self.targets == i)[0]
                
                if i in self.labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            # print(indexs)
            self.data = self.data[indexs]
            self.targets = self.targets[indexs]                   
            self.omic = self.omic[indexs]        
            self.batch = self.batch[indexs]  
            
            if self.transform is not None:
                self.dataoe = self.dataoe[indexs]

    def __getitem__(self, index):
        if self.transform is not None:
            emb = self.dataoe[index]
        else:
            emb = self.data[index]
        target = self.targets[index]
        omic = self.omic[index]           
        batch = self.batch[index]  
        barcode = self.barcodes[index] 
        # print("self.data",emb)
        
        # print("self.targets",target)
        # print('embedding',emb)
        # print('target',target)
        return emb, target,omic,barcode,batch #omic不参与训练只用于验证时判断对atac细胞的准确率
    def __len__(self):
            return len(self.data)
# 有trans
class RNA_UNCR(Dataset):
    def __init__(self,  data, targets,omic,batch=None,indexs=None,dataoe=None, train=True,
                 transform=None, target_transform=None):
        super().__init__()
        self.data = torch.from_numpy(data).float()
        self.data = torch.log(self.data + 1.)
        self.targets = np.array(targets)
        self.omic = np.array(omic)         
        if batch is not None:
            self.batch = np.array(batch)    
        self.transform = transform
        self.target_transform = target_transform
        self.dataoe = dataoe 
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = self.targets[indexs]
            self.indexs = indexs
            
            self.omic = self.omic[indexs]    
            self.batch = self.batch[indexs]  
            if self.transform is not None:
                self.dataoe = self.dataoe[indexs]
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        if self.transform is not None:
            emb = self.dataoe[index]
        else:   
            emb = self.data[index]
            
        target = self.targets[index]        
        omic = self.omic[index]
        batch = self.batch[index]   
        return emb, target, self.indexs[index], omic,batch 
    def __len__(self):
            return len(self.data) 
        
        

class ATAC(Dataset):#提供 coembeddding orembedding 标签 不确定性
    def __init__(self, data,targets,omic,barcodes,batch=None, indexs=None, dataoe=None,temperature=None, temp_uncr=None,transform=None, train=True,
                  target_transform=None, batch_num = None, no_class = None):
        super().__init__()
        
        self.targets = np.array(targets)
        # no_seen = self.targets.max()
        self.data = torch.from_numpy(data).float()
        self.targets=torch.zeros(self.targets.shape[0], no_class).scatter_(1, torch.tensor(self.targets).view(-1, 1).long(), 1)
        self.transform = transform
        self.barcodes = barcodes
        self.target_transform = target_transform
        self.omic = np.array(omic) 
        self.omic = torch.zeros(self.omic.shape[0], 2).scatter_(1, torch.tensor(self.omic).view(-1,1).long(), 1)
        if batch is not None:
            self.batch = np.array(batch)  
            self.batch = torch.zeros(self.batch.shape[0], batch_num).scatter_(1, torch.tensor(self.batch).view(-1,1).long(), 1)
        self.dataoe = dataoe  
        if temperature is not None:
            self.temp = temperature*np.ones(len(self.targets))
        else:
            self.temp = np.ones(len(self.targets))


        if temp_uncr is not None:
            self.temp[temp_uncr['index']] = temp_uncr['uncr']

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            if self.transform is not None:
                self.dataoe = self.dataoe[indexs]
            # print("self.data",self.data)
            self.targets = self.targets[indexs]
            # print("self.targets",self.targets)
            self.temp = self.temp[indexs]
            self.omic = self.omic[indexs]    
            self.batch = self.batch[indexs]  
            self.barcodes = self.barcodes[indexs] 
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))
    def __len__(self):
            return len(self.data)
    def __getitem__(self, index):
        if self.transform is not None:
            emb = self.dataoe[index]
        else:
            emb = self.data[index]
        
        target = self.targets[index]
        omic = self.omic[index]   
        batch = self.batch[index]   
        barcodes = self.barcodes[index] 
        return emb, target, self.indexs[index], self.temp[index],omic,barcodes,batch #omic不参与训练只用于验证时判断对atac细胞的准确率
    


class ATAC_TEST(Dataset):
    def __init__(self, data,targets,omic,barcodes,batch=None,dataoe=None,train=False,
                 transform=None, target_transform=None,
                 labeled_set=None, batch_num = None, no_class = None):
        super().__init__()
        self.targets = np.array(targets)
        # print(self.targets)        
        self.data = torch.from_numpy(data).float()
        self.targets = torch.zeros(self.targets.shape[0], no_class).scatter_(1, torch.tensor(self.targets).view(-1,1).long(), 1) 
        self.barcodes = barcodes
        self.transform = transform
        self.target_transform = target_transform
        self.omic = np.array(omic) 
        self.omic = torch.zeros(self.omic.shape[0], 2).scatter_(1, torch.tensor(self.omic).view(-1,1).long(), 1)
        if batch is not None: 
            self.batch = np.array(batch)  
            self.batch = torch.zeros(self.batch.shape[0], batch_num).scatter_(1, torch.tensor(self.batch).view(-1,1).long(), 1)

        self.dataoe = dataoe  
        # print(len(labeled_set))
        indexs = []
        if labeled_set is not None:
            self.labeled_set = labeled_set
            # print(labeled_set)
            # self.labeled_set=map_names_to_indices(labeled_set,category_to_index)
            # print(self.labeled_set)
            for i in range(no_class):#
                idx = np.where(self.targets == i)[0]
                
                if i in self.labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            # print(indexs)
            self.data = self.data[indexs]
            self.targets = self.targets[indexs]                   
            self.omic = self.omic[indexs]        
            self.batch = self.batch[indexs]  
            self.barcodes = self.barcodes[indexs]  
            if self.transform is not None:
                self.dataoe = self.dataoe[indexs]

    def __getitem__(self, index):
        if self.transform is not None:
            emb = self.dataoe[index]
        else:
            emb = self.data[index]
        target = self.targets[index]
        omic = self.omic[index]           
        batch = self.batch[index]  
        barcodes = self.barcodes[index] 
        # print("self.data",emb)
        # print("self.targets",target)
        
        # print('embedding',emb)
        # print('target',target)
        return emb, target,omic,barcodes,batch #omic不参与训练只用于验证时判断对atac细胞的准确率
    def __len__(self):
            return len(self.data)