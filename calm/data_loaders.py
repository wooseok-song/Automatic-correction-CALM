"""
CALM
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import munch
import os
import torch


from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split
from skimage import io, transform  # 이미지 I/O와 변형을 위해 필요한 library

_IMAGE_MEAN_VALUE = [0.485, 0.456, 0.406]
_IMAGE_STD_VALUE = [0.229, 0.224, 0.225]
_SPLITS = ('train', 'val', 'test')


class MyDataset(torch.utils.data.Dataset):  # My customize dataset
    def __init__(self, data_path, label, phase, transform=None):
        self.label = label
        self.data_path = data_path
        self.phase = phase
        self.transform = transform
        self.dirlist = os.listdir(self.data_path + str(self.phase))

    def __len__(self):
        return len(self.dirlist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = os.path.join(self.data_path, str(self.phase), self.dirlist[idx])
        image = io.imread(image_name)
        label = self.label
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return image, label, idx


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def dataloader_wrapping(loaders, batch_size, workers):
    for split in _SPLITS:
        loaders[split] = DataLoader(loaders[split],
                                    batch_size=batch_size,
                                    shuffle=split in ['train', 'val'],
                                    num_workers=workers)


def configure_metadata(metadata_root):
    metadata = mch()
    metadata.image_ids = os.path.join(metadata_root, 'image_ids.txt')
    metadata.image_ids_proxy = os.path.join(metadata_root,
                                            'image_ids_proxy.txt')
    metadata.class_labels = os.path.join(metadata_root, 'class_labels.txt')
    metadata.image_sizes = os.path.join(metadata_root, 'image_sizes.txt')
    metadata.localization = os.path.join(metadata_root, 'localization.txt')
    return metadata


def get_image_ids(metadata, proxy=False):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    image_ids = []
    suffix = '_proxy' if proxy else ''
    with open(metadata['image_ids' + suffix]) as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))
    return image_ids


def get_class_labels(metadata):
    """
    class_labels.txt has the structure

    <path>,<integer_class_label>
    path/to/image1.jpg,0
    path/to/image2.jpg,1
    path/to/image3.jpg,1
    ...
    """
    class_labels = {}
    with open(metadata.class_labels) as f:
        for line in f.readlines():
            image_id, class_label_string = line.strip('\n').split(',')
            class_labels[image_id] = int(class_label_string)
    return class_labels


def get_mask_paths(metadata):
    """
    localization.txt (for masks) has the structure

    <path>,<link_to_mask_file>,<link_to_ignore_mask_file>
    path/to/image1.jpg,path/to/mask1a.png,path/to/ignore1.png
    path/to/image1.jpg,path/to/mask1b.png,
    path/to/image2.jpg,path/to/mask2a.png,path/to/ignore2.png
    path/to/image3.jpg,path/to/mask3a.png,path/to/ignore3.png
    ...

    One image may contain multiple masks (multiple mask paths for same image).
    One image contains only one ignore mask.
    """
    mask_paths = {}
    ignore_paths = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, mask_path, ignore_path = line.strip('\n').split(',')
            if image_id in mask_paths:
                mask_paths[image_id].append(mask_path)
                assert (len(ignore_path) == 0)
            else:
                mask_paths[image_id] = [mask_path]
                ignore_paths[image_id] = ignore_path
    return mask_paths, ignore_paths


class ImageLabelDataset(Dataset):
    def __init__(self, data_root, metadata_root, transform, proxy,
                 superclass_labels=None):
        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root)
        self.transform = transform
        self.image_ids = get_image_ids(self.metadata, proxy=proxy)
        self.image_labels = get_class_labels(self.metadata)

        if superclass_labels:
            self.image_ids = [
                image_id for image_id in self.image_ids \
                if self.image_labels[image_id] in superclass_labels
            ]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        image = Image.open(os.path.join(self.data_root, image_id))
        image = image.convert('RGB')
        image = self.transform(image)
        return image, image_label, image_id

    def __len__(self):
        return len(self.image_ids)


def get_data_loader(data_roots, metadata_root, batch_size, workers,
                    resize_size, crop_size, data_path, proxy_training_set,
                    superclass_labels=None):
    dataset_transforms = dict(
        train=transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        val=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]),
        test=transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGE_MEAN_VALUE, _IMAGE_STD_VALUE)
        ]))

    loaders = dict()

    real = MyDataset(data_path, 0, '0.real', transform=dataset_transforms['train'])
    normal = MyDataset(data_path, 1, '1.normal', transform=dataset_transforms['train'])
    artifact = MyDataset(data_path, 2, '2.artifact', transform=dataset_transforms['train'])

    dataset = real + normal + artifact

    loaders['train'], loaders['test'] = train_test_split(dataset, test_size=0.2, shuffle=True)
    loaders['train'], loaders['val'] = train_test_split(loaders['train'], test_size=0.3, shuffle=True)
    dataloader_wrapping(loaders, batch_size, 0)

    return loaders
