from os.path import join

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch

from src.datasets.common.dataset_skeleton import DatasetSkeleton
# IMPLEMENTATION of DivideMix is based on the official repo https://github.com/LiJunnan1992/DivideMix
# We used the Clothing1M as baseline

class benchmark_dataset(Dataset):
    def __init__(self, ds, dataset_info, transform, mode, num_samples=0, pred=[], probability=[], paths=[], num_class=14):
        
        self.ds = ds
        self.dataset_info = dataset_info
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}

        dataset_root, dataset_name = dataset_info.raw_data_root_directory, dataset_info.name

        mode = 'hard'
        paths_train, gt_train = ds.get_training_subsets('train', mode)
        paths_val, gt_val = ds.get_training_subsets('val', mode)
        # paths_test, gt_test = ds.get_training_subsets('test', mode)


        # cast gt from one hot encoded to single value
        gt_train = np.argmax(gt_train,axis=1)
        gt_val = np.argmax(gt_val,axis=1)
        # gt_test = np.argmax(gt_test,axis=1)

        # cast paths to full paths names
        paths_train = [join(dataset_root, path) for path in paths_train]
        paths_val = [join(dataset_root, path) for path in paths_val]


        self.train_labels = {p:l for p,l in zip(paths_train,gt_train)}
        self.test_labels = {p:l for p,l in zip(paths_val,gt_val)}

        self.val_imgs = paths_val


        if self.mode == 'all':

            self.train_imgs = paths_train
        elif self.mode == "labeled":   
            train_imgs = paths 
            pred_idx = pred.nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
        elif self.mode == "unlabeled":  
            train_imgs = paths 
            pred_idx = (1-pred).nonzero()[0]  
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
        elif self.mode == 'complete':
            print(f"Init Data loader all images")
            paths_all, _ = ds.get_training_subsets('all', mode)
            relative_all = paths_all
            paths_all = [join(dataset_root, path) for path in paths_all]
            self.train_imgs = paths_all
            self.relative = relative_all
            print(f"Found {len(paths_all)} images")



    def __getitem__(self, index):  
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2
        elif self.mode=='complete':
            img_path = self.train_imgs[index]
            rel_path = self.relative[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            return img1, rel_path
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, img_path
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target    
        
    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)            
        
class benchmark_loader():
    def __init__(self, ds, dataset_info : DatasetSkeleton, batch_size, num_workers): #num_batches
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = 0
        self.ds = ds
        self.dataset_info = dataset_info

        mean = (dataset_info.channel_mean[0],dataset_info.channel_mean[1],dataset_info.channel_mean[2])
        std = (dataset_info.channel_std[0],dataset_info.channel_std[1],dataset_info.channel_std[2])

        input_size = dataset_info.input_size
        if input_size < 50:
            input_size *= 2 # increase slightly to ensure large enough for network

        self.transform_train = transforms.Compose([
                transforms.Resize(input_size),
                transforms.RandomCrop(int(input_size*0.9)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize(mean,std),
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(int(input_size*0.9)),
                transforms.ToTensor(),
                transforms.Normalize(mean,std),
            ])        
    def run(self,mode,pred=[],prob=[],paths=[]):        
        if mode=='warmup':
            warmup_dataset = benchmark_dataset(self.ds, self.dataset_info,transform=self.transform_train, mode='all',num_samples=self.num_batches*self.batch_size*2)
            warmup_loader = DataLoader(
                dataset=warmup_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)  
            return warmup_loader
        elif mode=='train':
            labeled_dataset = benchmark_dataset(self.ds, self.dataset_info,transform=self.transform_train, mode='labeled',pred=pred, probability=prob,paths=paths)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)           
            unlabeled_dataset = benchmark_dataset(self.ds, self.dataset_info,transform=self.transform_train, mode='unlabeled',pred=pred, probability=prob,paths=paths)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_loader,unlabeled_loader
        elif mode=='eval_train':
            eval_dataset = benchmark_dataset(self.ds, self.dataset_info,transform=self.transform_test, mode='all',num_samples=self.num_batches*self.batch_size)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader
        elif mode=='val':
            val_dataset = benchmark_dataset(self.ds, self.dataset_info,transform=self.transform_test, mode='val')
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)             
            return val_loader
        elif mode=='complete':
            complete_dataset = benchmark_dataset(self.ds, self.dataset_info,transform=self.transform_test, mode='complete')
            complete_loader = DataLoader(
                dataset=complete_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return complete_loader

        raise ValueError(f'mode {mode} not implemented')