import os
import torch

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from .samplers import RASampler
import utils

import numpy as np
import torchvision
from torch.utils.data import  Subset

class CustomVisionDataset(torchvision.datasets.VisionDataset):
    def __init__(self, tensor_dataset, transform=None, target_transform=None):
        super(CustomVisionDataset, self).__init__(root=None, transform=transform, target_transform=target_transform)
        self.tensor_dataset = tensor_dataset

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, index):
        sample, target = self.tensor_dataset[index]
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target


def dataloader(args):
    model_type = args.model.split("_")[0]
    if model_type == "deit" or "swin":
        if 'imagenet_c' in args.data_set:
            data_set_split = args.data_set.split('-')
            corruption_type = data_set_split[-1]
            train_n = int(data_set_split[-2])
            severity = data_set_split[-3]

            data_root = args.data
            image_dir = os.path.join(data_root, 'imagenet-c', corruption_type, str(severity))
            # dataset = ImageFolder(image_dir, transform=transforms.ToTensor())
            dataset = ImageFolder(image_dir)
            indices = list(range(len(dataset.imgs))) #50k examples --> 50 per class
            assert train_n <= 20000
            labels = {}
            y_corr = dataset.targets
            for i in range(max(y_corr)+1):
                labels[i] = [ind for ind, n in enumerate(y_corr) if n == i] 
            num_ex = train_n // (max(y_corr)+1)
            tr_idxs = []
            val_idxs = []
            test_idxs = []
            for i in range(len(labels.keys())):
                np.random.shuffle(labels[i])
                tr_idxs.append(labels[i][:num_ex])
                val_idxs.append(labels[i][num_ex:num_ex+10])
                # tr_idxs.append(labels[i][:num_ex+10])
                test_idxs.append(labels[i][num_ex+10:num_ex+20])
            tr_idxs = np.concatenate(tr_idxs)
            val_idxs = np.concatenate(val_idxs)
            test_idxs = np.concatenate(test_idxs)

            dataset_train = CustomVisionDataset(Subset(dataset, tr_idxs), transform=build_transform(True, args))
            dataset_val = CustomVisionDataset(Subset(dataset, val_idxs), transform=build_transform(False, args))
            dataset_test = CustomVisionDataset(Subset(dataset, test_idxs), transform=build_transform(False, args))
        else:
            dataset_train = build_dataset(is_train=True, args=args)
            dataset_val = build_dataset(is_train=False, args=args)

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        # Data
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        raise NotImplementedError

    return data_loader_train, data_loader_val


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    else:
        raise NotImplementedError

    return dataset


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
