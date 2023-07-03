from os.path import join
import string
from cv2 import transform

import numpy as np
#import tensorflow as tf
#from tensorflow.keras import layers
#from tensorflow.keras.applications import imagenet_utils
#from tensorflow.keras import Model
#from tensorflow.keras.applications.densenet import DenseNet121
#from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
#from tensorflow.keras.applications.resnet_v2 import ResNet50V2
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from src.util.json import DatasetDCICJson
from src.util.const import get_all_dataset_infos

from PIL import Image



def torch_get_model(dataset_name, num_classes, weights='imagenet', network_name="resnet50v2", dropout=0.5):
    """
    get the model for the given network name
    :return:
    """

    #  parameters  
    dataset_info = get_all_dataset_infos()[dataset_name]
    w_h = dataset_info.get_input_size()

    # get backbone

    if network_name == "resnet50v2":

        pretrained = True
        # adjust input
        backbone = torchvision.models.resnet50(pretrained=pretrained) # pretrained=False just for debug reasons
        # adjust output
        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise ValueError("%s is not an allowed network name" % network_name)
    return backbone



class TDataModule(pl.LightningDataModule):

    class TDataset(torch.utils.data.Dataset):
        """
        Minimal pytorch dataset
        """
        def __init__(self, inputs, labels, transform = None):
            super().__init__()
            self.inputs = inputs
            self.labels = labels
            self.transform = transform
        def __len__(self):
            return len(self.inputs)
        def __getitem__(self, index):
            x = self.inputs[index]
            y = self.labels[index]
            # force float Tensors
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
                x.type(torch.float32)
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y)
                y.type(torch.float32)
            if isinstance(x, list):
                x = torch.Tensor(x)
                x.type(torch.float32)
            if isinstance(y, list):
                y = torch.Tensor(y)
                y.type(torch.float32)
            x,y=x.type(torch.cuda.FloatTensor),y.type(torch.cuda.FloatTensor)
            return x, y


    def __init__(self,datasetDCIC: DatasetDCICJson, dataset_info, batch_size=2, train_transform = None, val_transform = None, custom_collate = None):
        super().__init__()
        self.custom_collate = custom_collate
        self.train_transform = train_transform
        self.val_transform = val_transform

        # get numpy from dataset
        self.images_ds_train, self.labels_ds_train, _, self.soft_gt_train = TDataModule.tensor_get_training_subsets(datasetDCIC, 'train', dataset_info.raw_data_root_directory, dataset_info.name, extra_transform = train_transform)

        self.images_ds_val, self.labels_ds_val, _, self.soft_gt_val = TDataModule.tensor_get_training_subsets(datasetDCIC, 'val', dataset_info.raw_data_root_directory, dataset_info.name, extra_transform = val_transform)

        self.images_ds_all, _, _, _ = TDataModule.tensor_get_training_subsets(datasetDCIC, 'all', dataset_info.raw_data_root_directory, dataset_info.name)

        # build pytorch datasets
        self.trainDataset = TDataModule.TDataset(self.images_ds_train, self.labels_ds_train, train_transform)
        self.valDataset = TDataModule.TDataset(self.images_ds_val, self.labels_ds_val, val_transform)

        self.batch_size = batch_size
        self.num_samples = len(self.images_ds_train)

    def setup(self, stage = None):
        pass    

    def train_dataloader(self):
        # Return DataLoader for Training Data here
        return DataLoader(self.trainDataset, batch_size = self.batch_size, collate_fn = self.custom_collate)
    
    def val_dataloader(self):
        # Return DataLoader for Validation Data here
        return DataLoader(self.valDataset, batch_size = self.batch_size , collate_fn = self.custom_collate)
    
    def test_dataloader(self):
        return DataLoader(self.images_ds_all, batch_size = self.batch_size)
        # Return DataLoader for Testing Data here
        


    def tensor_get_training_subsets(datasetDCIC: DatasetDCICJson, _split, dataset_root, dataset_name, extra_transform = None):
        """
        Get a tensorflow dataset for the given dataset subset
        :param dataset_json:
        :param split: the desired data split or all
        :param dataset_name: used to determine the loading parameters of the iamges
        :return:
        """

        # filter for split and unknown gt class
        paths = np.array([
            path
            for i, (path, split, hard_gt, soft_gt) in enumerate(datasetDCIC.get_image_iterator())
            if split == _split or _split == 'all'
        ])

        path_full = np.array([
            join(dataset_root, path)
            for i, (path, split, hard_gt, soft_gt) in enumerate(datasetDCIC.get_image_iterator())
            if split == _split or _split == 'all'
        ])

        soft_gt = np.array([
            soft_gt
            for i, (path, split, hard_gt, soft_gt) in enumerate(datasetDCIC.get_image_iterator())
            if split == _split # soft gt does not exists for all
        ])

        list_ds = list(path_full)
        if len(paths) > 0:
            images_ds = list(map(TDataModule.get_parsing_function(dataset_name, extra_transform), list_ds))
        else:
            images_ds = list_ds  # no images loadable


        labels_ds = soft_gt

        return images_ds, labels_ds, paths, soft_gt

    def get_parsing_function(dataset_name,extra_transforms = None):
        """
        get the parsing function of the input depending on the input

        :param dataset_name:
        :return:
        """
        # get input size depending on dataset name

        dataset_info = get_all_dataset_infos()[dataset_name]
        input_size = dataset_info.get_input_size()
    # https://github.com/qubvel/segmentation_models.pytorch/issues/371


        tfms = transforms.Compose([ 
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        no_tensor_tfms = transforms.Compose([ 
            transforms.Resize(input_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    
        def parse_image(filename):
            """
            parse the image but apply no preprocessing
            :param filename:
            :return:
            """
            image = Image.open(filename)
            if extra_transforms:
                image = extra_transforms(image)
                if len(image) > 1:
                    image = [no_tensor_tfms(e)for e in image]
                else:
                    image = extra_transforms(image)
            else:
                image = tfms(image)
            return image

        return parse_image