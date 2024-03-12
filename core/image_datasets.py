import os
import math
import h5py
import torch
import random
import numpy as np
import pandas as pd
import blobfile as bf

from os import path as osp
from PIL import Image
from mpi4py import MPI
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# ImageFolder dataloader
# ============================================================================


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# ============================================================================
# CheXpert dataloader
# ============================================================================


def load_data_chexpert(
    *,
    data_dir,
    batch_size,
    image_size,
    partition='train',
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = CheXpertDataset(
        image_size,
        data_dir,
        partition,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        class_cond=class_cond,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=5, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True
        )
    while True:
        yield from loader


def load_data_pe(
    *,
    data_dir,
    batch_size,
    image_size,
    partition='train',
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = PEDataset(
        image_size,
        data_dir,
        partition,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        class_cond=class_cond,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=5, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True
        )
    while True:
        yield from loader

class CheXpertDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
    ):
        partition_df = pd.read_csv(osp.join(data_dir, 'list_eval_partition.csv'))
        self.data_dir = data_dir
        data = pd.read_csv(osp.join(data_dir, 'list_attr_chexpert.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

        self.query = query_label
        self.class_cond = class_cond

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        labels = sample[6:].to_numpy()
        
        #print('Query Disease:', self.data.columns[6:][self.query])
        if self.query != -1:
            labels = int(labels[self.query])
        else:
            labels = torch.from_numpy(labels.astype('float32'))
        img_file = sample['image_id']

        with open(osp.join(self.data_dir, 'img_chexpert', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)

        if self.query != -1:
            return img, labels

        if self.class_cond:
            return img, {'y': labels}
        else:
            return img, {}


class PEDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        path='img_pe',
        task='classification', # it can either be classification or detection
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        normalize=True,
    ):
        self.data_dir = data_dir
        data = pd.read_csv(osp.join(data_dir, 'list_attr_pe.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[data['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

        self.class_cond = class_cond
        self.task = task
        self.path = path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        img_file = sample['Path']

        # Determine the label based on the task
        if self.task == 'classification':
            # Convert 'Healthy/Unhealthy' to binary labels
            label = 0 if sample['group'] in [0, 2] else 1
        elif self.task == 'detection':
            # Binary label for dot detection based on 'group'
            label = 1 if sample['group'] in [0, 1] else 0
        elif self.task == 'both':
            det_label = 1 if sample['group'] in [0, 1] else 0
            cls_label = 0 if sample['group'] in [0, 2] else 1
        else:
            raise ValueError(f'Unknown task {self.task}')

        # Rest of the code to load and transform the image...
        with open(os.path.join(self.data_dir, 'img_pe', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)
        if self.task == 'both':
            return img, {'y': cls_label, 'z': det_label}
        return img, label




class PE90DotNoSupportDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        path='imgs',
        task='classification', # it can either be classification or detection
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        normalize=True,
        biased=False
    ):
        self.data_dir = data_dir
        data = pd.read_csv(osp.join(data_dir, 'info.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[data['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

        self.class_cond = class_cond
        self.task = task
        self.path = path
        if biased:
            self.data = self.data[self.data['group'].isin([0, 3])]       # self.data.replace(-1, 0, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        img_file = sample['Path']

        # Determine the label based on the task
        if self.task == 'classification':
                # Convert 'Healthy/Unhealthy' to binary labels make sure to change back to [0, 2]
                label = 0 if sample['group'] in [0, 2] else 1 
        elif self.task == 'detection':
            # Binary label for dot detection based on 'group'
            label = 1 if sample['group'] in [0, 1] else 0
        elif self.task == 'both':
            det_label = 1 if sample['group'] in [0, 1] else 0
            cls_label = 0 if sample['group'] in [0, 2] else 1 
        else:
            raise ValueError(f'Unknown task {self.task}')
        
        # Rest of the code to load and transform the image...
        with open(os.path.join(self.data_dir, self.path, img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)
        if self.task == 'both':
            return img, {'y': cls_label, 'z': det_label}
        return img, label


class MedicalDevicePEDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        csv_dir,
        partition,
        path='img_chexpert',
        task='classification', # it can either be classification or detection
        shard=0,
        num_shards=1,
        ratio=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        normalize=True,
        biased=False,
        rebalance=True,
        sample=False
    ):
        self.data_dir = data_dir
        if csv_dir is None:
            raise ValueError("unspecified csv directory")
        data = pd.read_csv(osp.join(csv_dir, 'list_attr_md.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[data['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

        self.class_cond = class_cond
        self.task = task
        self.path = path
        self.biased = biased
        if rebalance:
            self.data = self._balance_subjects(self.data, ratio=ratio, sample=sample)
    @staticmethod
    def _balance_subjects(df, ratio=0.1, sample=False):
        if ratio==1:
            return df
        else:
            if sample:
                gr_0 = df[df['group']==0].sample(frac=0.1)
                gr_3 = df[df['group']==3].sample(frac=0.1)
            else:
                gr_0 = df[df['group']==0]
                gr_3 = df[df['group']==3]
            majority_size = min(len(gr_0), len(gr_3))
            gr_0 = gr_0[:majority_size]
            gr_3 = gr_3[:majority_size]
            r1 = int(ratio * majority_size)
            r2 = int(ratio * majority_size)
            gr_1 = df[df['group']==1][:r2]
            gr_2 = df[df['group']==2][:r1]
            print(len(gr_0), len(gr_1), len(gr_2), len(gr_3))
            return pd.concat([gr_0, gr_1, gr_2, gr_3],axis=0)   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        img_file = sample['Path']

        # Determine the label based on the task
        if self.task == 'classification':
                # Convert 'Healthy/Unhealthy' to binary labels make sure to change back to [0, 2]
                label = 0 if sample['group'] in [0, 2] else 1
        elif self.task == 'detection':
            # Binary label for dot detection based on 'group'
            label = 1 if sample['group'] in [0, 1] else 0
        elif self.task == 'both':
            det_label = 1 if sample['group'] in [0, 1] else 0
            cls_label = 0 if sample['group'] in [0, 2] else 1
        else:
            raise ValueError(f'Unknown task {self.task}')

        # Rest of the code to load and transform the image...
        with open(os.path.join(self.data_dir, self.path, img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)
        if self.task == 'both':
            return img, {'y': cls_label, 'z': det_label}
        return img, label
    