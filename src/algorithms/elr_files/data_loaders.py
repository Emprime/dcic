import sys
from os.path import join

from torchvision import datasets, transforms
import torch
from PIL import Image
import random
from typing import Tuple, Union, Optional
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from src.datasets.common.dataset_skeleton import DatasetSkeleton


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    valid_sampler: Optional[SubsetRandomSampler]
    sampler: Optional[SubsetRandomSampler]

    def __init__(self, train_dataset, batch_size, shuffle, validation_split: float, num_workers, pin_memory,
                 collate_fn=default_collate, val_dataset=None):
        self.collate_fn = collate_fn
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.val_dataset = val_dataset

        self.batch_idx = 0
        self.n_samples = len(train_dataset) if val_dataset is None else len(train_dataset) + len(val_dataset)
        self.init_kwargs = {
            'dataset': train_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }
        if val_dataset is None:
            self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
            super().__init__(sampler=self.sampler, **self.init_kwargs)
        else:
            super().__init__(**self.init_kwargs)

    def _split_sampler(self, split) -> Union[Tuple[None, None], Tuple[SubsetRandomSampler, SubsetRandomSampler]]:
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        print(f"Train: {len(train_sampler)} Val: {len(valid_sampler)}")

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self, bs = 1000):
        if self.val_dataset is not None:
            kwargs = {
                'dataset': self.val_dataset,
                'batch_size': bs,
                'shuffle': False,
                'collate_fn': self.collate_fn,
                'num_workers': self.num_workers
            }
            return DataLoader(**kwargs)
        else:
            print('Using sampler to split!')
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


class AdvancedDataLoader(BaseDataLoader):
    def __init__(self, ds, dataset_info : DatasetSkeleton, batch_size, shuffle=True, validation_split=0.0, num_batches=0, training=True, num_workers=4, pin_memory=True):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.training = training
        self.ds = ds

        self.dataset_info = dataset_info

        mean = (self.dataset_info.channel_mean[0], self.dataset_info.channel_mean[1], self.dataset_info.channel_mean[2])
        std = (self.dataset_info.channel_std[0], self.dataset_info.channel_std[1], self.dataset_info.channel_std[2])

        input_size = self.dataset_info.input_size
        if input_size < 50:
            input_size *= 2  # increase slightly to ensure large enough for network

        self.transform_train = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop(int(input_size * 0.9)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.transform_val = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(int(input_size * 0.9)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.train_dataset, self.val_dataset = get_datasets(ds, dataset_info, num_samples=self.num_batches*self.batch_size, train=training,
                transform_train=self.transform_train, transform_val=self.transform_val)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)


def get_datasets(ds, dataset_info, num_samples=0, train=True,
                 transform_train=None, transform_val=None):
    if train:
        train_dataset = Benchmark(ds,dataset_info, num_samples=num_samples, train=train, transform=transform_train)
        val_dataset = Benchmark(ds,dataset_info, val=train, transform=transform_val)
        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")

    else:
        train_dataset = []
        val_dataset = Benchmark(ds,dataset_info, test=(not train), transform=transform_val)
        print(f"Test: {len(val_dataset)}")

    return train_dataset, val_dataset


class Benchmark(torch.utils.data.Dataset):

    def __init__(self, ds,dataset_info:DatasetSkeleton, num_samples=0, train=False, val=False, test=False, transform=None):
        self.ds = ds
        self.dataset_info = dataset_info
        self.transform = transform
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}

        self.train = train
        self.val = val
        self.test = test

        num_class = self.dataset_info.num_classes
        dataset_root, dataset_name = dataset_info.raw_data_root_directory, dataset_info.name

        mode = 'hard'
        paths_train, gt_train = ds.get_training_subsets('train', mode)
        paths_val, gt_val = ds.get_training_subsets('val', mode)

        # cast gt from one hot encoded to single value
        gt_train = np.argmax(gt_train, axis=1)
        gt_val = np.argmax(gt_val, axis=1)
        # gt_test = np.argmax(gt_test,axis=1)

        # cast paths to full paths names
        paths_train = [join(dataset_root, path) for path in paths_train]
        paths_val = [join(dataset_root, path) for path in paths_val]

        self.train_labels = {p: l for p, l in zip(paths_train, gt_train)}
        self.test_labels = {p: l for p, l in zip(paths_val, gt_val)}

        self.val_imgs = paths_val


        if train:
            train_imgs = [(i,path) for i, path in enumerate(paths_train)]
            self.num_raw_example = len(train_imgs)
            num_samples = self.num_raw_example
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for id_raw, impath in train_imgs:
                # label = self.train_labels[impath]
                # if class_num[label] < (num_samples / num_class) and len(self.train_imgs) < num_samples:
                self.train_imgs.append((id_raw, impath))
                    # class_num[label] += 1
            random.shuffle(self.train_imgs)

        elif test:
            print(f"Init Data loader all images")
            paths_all, _ = ds.get_training_subsets('all', mode)
            relative_all = paths_all
            paths_all = [join(dataset_root, path) for path in paths_all]
            self.test_imgs = paths_all
            self.relative = relative_all
            print(f"Found {len(paths_all)} images")


        elif val:
            self.val_imgs = paths_val

    def __getitem__(self, index):
        if self.train:
            id_raw, img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
        elif self.val:
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]
        elif self.test:
            img_path = self.test_imgs[index]
            target = 0 #self.test_labels[img_path]
        image = Image.open(img_path).convert('RGB')
        if self.train:
            img0 = self.transform(image)

        if self.test:
            img = self.transform(image)
            return img, target, self.relative[index], target
        elif self.val:
            img = self.transform(image)
            return img, target, index, target
        else:
            return img0, target, id_raw, target

    def __len__(self):
        if self.test:
            return len(self.test_imgs)
        if self.val:
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)

    def flist_reader(self, flist):
        imlist = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(" ")
                impath = self.root + row[0]
                imlabel = float(row[1].replace('\n', ''))
                imlist.append((impath, int(imlabel)))
        return imlist


