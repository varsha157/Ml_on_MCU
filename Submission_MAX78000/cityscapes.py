import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from typing import Any, List, Tuple

import ai8x

class CityScapesDataset(Dataset):

    classes_dict = {"unlabeled": 0, "ego vehicle": 1, "rectification border": 2, "out of roi": 3, "static": 4, "dynamic": 5, 
                    "ground": 6, "road": 7, "sidewalk": 8, "parking": 9, "rail track": 10, "building": 11, "wall": 12, "fence": 13, 
                    "guard rail": 14, "bridge": 15, "tunnel": 16, "pole": 17, "polegroup": 18, "traffic light": 19, "traffic sign": 20, 
                    "vegetation": 21, "terrain": 22, "sky": 23, "person": 24, "rider": 25, "car": 26, "truck": 27, "bus": 28, 
                    "caravan": 29, "trailer": 30, "train": 31, "motorcycle": 32, "bicycle": 33, "license plate": -1}


    def __init__(self, root, split='train', transform=None) -> None:
        self.root = root
        self.split = split
        self.transform = transform

        self.classes = ['vegetation', 'car']

        self.images = []
        self.labels = []

        self.images_dir = os.path.join(root, 'leftImg8bit', split)
        self.labels_dir = os.path.join(root, 'gtFine', split)

        self.label_map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 2, 9: 0, 
                          10: 0, 11: 3, 12: 4, 13: 5, 14: 0, 15: 0, 16: 0, 17: 6, 18: 0, 
                          19:7, 20: 8, 21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16, 
                          29: 0, 30: 0, 31: 16, 32: 17, 33: 18, -1: -1}
                
        self.label_map_compressed = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 0, 
                                     10: 0, 11: 2, 12: 2, 13: 2, 14: 0, 15: 0, 16: 0, 17: 3, 18: 3, 
                                     19: 3, 20: 3, 21: 4, 22: 4, 23: 0, 24: 5, 25: 5, 26: 6, 27: 6, 28: 6, 
                                     29: 0, 30: 0, 31: 6, 32: 6, 33: 6, -1: -1}
        
        self.label_map_six = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 0, 
                                     10: 0, 11: 2, 12: 2, 13: 2, 14: 0, 15: 0, 16: 0, 17: 2, 18: 2, 
                                     19: 2, 20: 2, 21: 3, 22: 3, 23: 1, 24: 4, 25: 4, 26: 5, 27: 5, 28: 5, 
                                     29: 0, 30: 0, 31: 5, 32: 5, 33: 5, -1: -1}
        
        self.label_map_four = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 0, 9: 0, 
                                     10: 0, 11: 1, 12: 1, 13: 1, 14: 0, 15: 0, 16: 0, 17: 1, 18: 0, 
                                     19: 1, 20: 1, 21: 2, 22: 2, 23: 0, 24: 3, 25: 3, 26: 3, 27: 3, 28: 3, 
                                     29: 0, 30: 0, 31: 3, 32: 3, 33: 3, -1: -1}
        
        self.label_map_three = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 
                                     10: 0, 11: 1, 12: 1, 13: 1, 14: 0, 15: 0, 16: 0, 17: 2, 18: 0, 
                                     19: 2, 20: 2, 21: 1, 22: 1, 23: 0, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 
                                     29: 0, 30: 0, 31: 2, 32: 2, 33: 2, -1: -1}
        
        self.label_map_single = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 
                                 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 
                                 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 1, 27: 1, 28: 1, 
                                 29: 0, 30: 0, 31: 1, 32: 1, 33: 1, -1: -1}
        
        self.crop = transforms.Compose([transforms.Resize((352, 352))])

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            lbl_dir = os.path.join(self.labels_dir, city)
            for file_name in os.listdir(img_dir):
                label_id = []
                label_name = "{}_{}".format(file_name.split("_leftImg8bit")[0], "gtFine_labelIds.png")
                label_id.append(os.path.join(lbl_dir, label_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.labels.append(label_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image = Image.open(self.images[idx]).convert("RGB")
        label = Image.open(self.labels[idx][0])

        label = np.array(label).astype(np.uint8)
        label = self.convert_label(label)
        #label = self.filter_classes(label)

        image = self.crop(image)
        label = self.crop(Image.fromarray(label))

        label = np.array(label).astype(np.int64)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
    def convert_label(self, label):
        for k, v in self.label_map_four.items():
            label[label == k] = v
        return label
    
    def filter_classes(self, label):
        initial_new_class_label = len(self.classes_dict) + 5
        new_class_label = initial_new_class_label
        for l_class in self.classes:
            if l_class not in self.classes_dict:
                print(f'Class is not in the data: {l_class}')
                return label
            
            label[(label == self.classes_dict[l_class])] = new_class_label
            new_class_label += 1
        
        label[(label < initial_new_class_label)] = new_class_label
        label -= initial_new_class_label

        return label
    

def get_cityscapes_dataset(data, load_train=True, load_test=True):

    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomInvert(p=0.1),
            transforms.RandomAutocontrast(p=0.1),
            transforms.RandomSolarize(0, p=0.1),
            transforms.Resize((80, 80)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = CityScapesDataset(os.path.join(data_dir, 'cityscapes'), split='train', transform=train_transform)

    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomInvert(p=0.1),
            transforms.RandomAutocontrast(p=0.1),
            transforms.RandomSolarize(0, p=0.1),
            transforms.Resize((80, 80)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = CityScapesDataset(os.path.join(data_dir, 'cityscapes'), split='val', transform=test_transform)

    else:
        test_dataset = None

    return train_dataset, test_dataset


def get_cityscapes_dataset_folded(data, load_train=True, load_test=True):

    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomInvert(p=0.1),
            transforms.RandomAutocontrast(p=0.1),
            transforms.RandomSolarize(0, p=0.1),
            transforms.ToTensor(),
            ai8x.normalize(args=args),
            ai8x.fold(fold_ratio=4)
        ])

        train_dataset = CityScapesDataset(os.path.join(data_dir, 'cityscapes'), split='train', transform=train_transform)

    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomInvert(p=0.1),
            transforms.RandomAutocontrast(p=0.1),
            transforms.RandomSolarize(0, p=0.1),
            transforms.ToTensor(),
            ai8x.normalize(args=args),
            ai8x.fold(fold_ratio=4)
        ])

        test_dataset = CityScapesDataset(os.path.join(data_dir, 'cityscapes'), split='val', transform=test_transform)

    else:
        test_dataset = None

    return train_dataset, test_dataset

datasets = [
    {
        'name': 'cityscapes',
        'input': (3, 80, 80),
        # 'output': ('unlabelled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic',
        #            'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence',
        #            'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign',
        #            'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
        #            'caravan', 'trailer', 'train', 'motorcycle', 'bicycle', 'license plate'),
        'output': (0,1,2,3),
        'weight': (1,1,1,1),
        'loader': get_cityscapes_dataset,
    },
    {
        'name': 'cityscapes_folded',
        'input': (48, 88, 88),
        'output': (0,1,2),
        'weight': (1,1,1),
        'loader': get_cityscapes_dataset_folded,
        'fold_ratio': 4,
    }
]

# Number of Model Weights: 278176
# Number of Model Bias: 908