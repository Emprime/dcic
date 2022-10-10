from os.path import join

from absl import app
from tqdm import tqdm

from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
# from pl_bolts.models.self_supervised.moco.moco2_module import Moco_v2
from src.algorithms.moco_module import Moco_v2
import numpy as np
from typing import Optional
from src.util.json import DatasetDCICJson
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter
import random
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner
import torch

class TDataset(Dataset):
        """
        Minimal pytorch dataset
        """

        def __init__(self, ds : DatasetDCICJson, dataset_info, split, transform, mode='soft'):
            super().__init__()
            self.dataset_info = dataset_info
            num_class = self.dataset_info.num_classes
            self.dataset_root, dataset_name = dataset_info.raw_data_root_directory, dataset_info.name
            self.split = split
            self.ds = ds

            if len(split.split("+")) == 1:
                self.paths, targets = self.ds.get_training_subsets(split, mode)
            else:
                # multiple datasets
                paths = []
                targets = []
                for s in split.split("+"):
                    p, t = self.ds.get_training_subsets(s, mode)
                    paths.append(p)
                    # hack to fake labels
                    if len(t) > 0 and len(t[0]) > 0:
                        targets.append(t)
                    else:
                        targets = []

                self.paths = np.array([el for p in paths for el in p])
                targets = np.array([el for t in targets for el in t])

            # cast gt from one hot encoded to single value
            if len(targets) > 0:
                # print(targets)
                targets = np.argmax(targets, axis=1)

            self.labels = {p: l for p, l in zip(self.paths, targets)}

            print(f"Created Dataset {self.split} with {len(self.paths)} images and {len(self.labels)} labels")


            if "train" in split:
                random.shuffle(self.paths)

            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            path = self.paths[index]
            complete_path = join(self.dataset_root,path)

            y = self.labels.get(path,0) # fake label if it does not exist

            image = Image.open(complete_path).convert('RGB')
            image = self.transform(image)

            if self.split != "all":
                # print("returned")
                return image, y
            else:
                return image, path

class BenchmarkDataModule(pl.LightningDataModule):


    def __init__(self, ds : DatasetDCICJson, dataset_info, batch_size,
                 num_workers: int = 8,
                 seed: int = 42,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = True,
                 use_unlabeled_data: bool = False,
                 ):

        """
                Args:

                    num_workers: how many workers to use for loading data
                    batch_size: the batch size
                    seed: random seed to be used for train/val/test splits
                    shuffle: If true shuffles the data every epoch
                    pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                                returning them
                    drop_last: If true drops the last incomplete batch
                """
        super().__init__()
        self.ds = ds
        self.dataset_info = dataset_info
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.use_unlabeled_data = use_unlabeled_data

        # check if batch size needs to be reduced for small dataset
        max_size = len(TDataset(self.ds, self.dataset_info, split='val', transform=self.train_transforms))
        batch_size = batch_size if batch_size < max_size else max_size
        self.batch_size = batch_size

        print("Used batch size", self.batch_size)

    def prepare_data(self):
        pass # nothing to do

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        if self.use_unlabeled_data:
            dataset = TDataset(self.ds, self.dataset_info,split='train+unlabeled',transform=self.train_transforms)
        else:
            dataset = TDataset(self.ds, self.dataset_info,split='train',transform=self.train_transforms)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        dataset = TDataset(self.ds, self.dataset_info, split='val', transform=self.val_transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        dataset = TDataset(self.ds, self.dataset_info, split='test', transform=self.val_transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory,
        )
        return loader

    def predict_dataloader(self):
        dataset = TDataset(self.ds, self.dataset_info, split='all', transform=self.val_transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory,
        )
        return loader


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class BenchmarkTransforms:

    def __init__(self, dataset_info,train,mode, severity : float = 0.5):
        """

        :param dataset_info: info about dataset
        :param train: boolean if training or evaluation
        :param mode: the type of ouput eg. single (finetune) or moco
        :param severity: value to increase or decrease the transform augmentation strength
        """
        self.train = train
        self.mode = mode
        mean = (dataset_info.channel_mean[0], dataset_info.channel_mean[1], dataset_info.channel_mean[2])
        std = (dataset_info.channel_std[0], dataset_info.channel_std[1], dataset_info.channel_std[2])


        assert severity >= 0 and severity <= 1, "Severity can only be a value between 0 and 1"

        jitter_strength = severity

        input_size = dataset_info.input_size

        self.color_jitter = transforms.ColorJitter(
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.8 * jitter_strength,
            0.2 * jitter_strength,
        )

        p_jitter = 0.8 * severity
        p_grayscale = 0.4 * severity
        p_guassian = 0 if severity < 0.25 else (0.25 if severity < 0.5 else 0.5)
        sigma_max = 0.1 if severity < 0.25 else (1 if severity < 0.5 else 2)
        crop_min = np.clip(1-severity, 0.3, 0.95)

        # easier transforms
        self.online_transform = \
            self.transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(input_size, scale=(crop_min, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )

        # image augmentation functions
        if self.train:
            if self.mode == 'finetune':
                # les tranforms for finetuning
                self.transform = self.online_transform
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(input_size, scale=(crop_min, 1.0)),
                        transforms.RandomApply([self.color_jitter], p=p_jitter),
                        transforms.RandomGrayscale(p=p_grayscale),
                        # transforms.RandomApply([GaussianBlur([0.1, sigma_max])], p=p_guassian), # TODO revert
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                    ]
                )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(int(input_size * 1.1)),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

        # special cases swav
        if self.mode == 'swav':

            size_crops = [224, 128] if dataset_info.input_size <= 112 else [192, 64]
            nmb_crops = [2, 1] if dataset_info.input_size <= 112 else [1, 2]
            min_scale_crops = [0.5, 0.3]
            max_scale_crops = [1.0, 0.5]


            transform = []

            for i in range(len(size_crops)):
                random_resized_crop = transforms.RandomResizedCrop(
                    size_crops[i],
                    scale=(min_scale_crops[i], max_scale_crops[i]),
                )

                transform.extend(
                    [
                        transforms.Compose(
                            [
                                random_resized_crop,
                                transforms.RandomApply([self.color_jitter], p=p_jitter),
                                transforms.RandomGrayscale(p=p_grayscale),
                                # transforms.RandomApply([GaussianBlur([0.1, sigma_max])], p=p_guassian),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std),
                            ]
                        )
                    ]
                    * nmb_crops[i]
                )

            self.transform = transform
            if train:
                self.transform.append(self.online_transform)
            else:
                self.transform.append(transforms.Compose(
                [
                    transforms.Resize(int(input_size * 1.1)),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ))

    def __call__(self, inp):

        if self.mode == 'moco':
            q = self.transform(inp)
            k = self.transform(inp)
            return q, k
        elif self.mode == 'simclr':
            q = self.transform(inp)
            k = self.transform(inp)
            l = self.online_transform(inp) # used for online Evaluation
            return q, k, l
        elif self.mode == 'swav':
            multi_crops = list(map(lambda transform: transform(inp), self.transform))
            return multi_crops
        elif self.mode == 'finetune':
            q = self.transform(inp)
            return q



