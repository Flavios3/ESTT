import os
from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr
from typing import List
from torchvision.transforms import ToTensor

class_map = {
   1: 13,  # ego_vehicle : vehicle
   7: 0,   # road
   8: 1,   # sidewalk
   11: 2,  # building
   12: 3,  # wall
   13: 4,  # fence
   17: 5,  # pole
   18: 5,  # poleGroup: pole
   19: 6,  # traffic light
   20: 7,  # traffic sign
   21: 8,  # vegetation
   22: 9,  # terrain
   23: 10,  # sky
   24: 11,  # person
   25: 12,  # rider
   26: 13,  # car : vehicle
   27: 13,  # truck : vehicle
   28: 13,  # bus : vehicle
   32: 14,  # motorcycle
   33: 15,  # bicycle
}

##############################################################################################################################################################################

class GTAVDataset(VisionDataset):

    def __init__(self,
                 root: str,
                 list_samples: List[str],
                 transform: tr.Compose = None):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.target_transform = self.get_mapping()
        self.style_augment = None

    @staticmethod
    def get_mapping():
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for k, v in class_map.items():
            mapping[k] = v
        return lambda x: from_numpy(mapping[x])
        
    def __getitem__(self, index: int) -> Any:

        sample_name = self.list_samples[index]
        img_path = os.path.join('data/GTAV+Cityscapes/data/GTA5/images', sample_name)
        img = Image.open(img_path).convert('RGB')     
        label_path = os.path.join('data/GTAV+Cityscapes/data/GTA5/labels', sample_name)
        label = Image.open(label_path).convert('P')
        
        if self.style_augment is not None:
            img = self.style_augment.apply_style(img)
        if self.transform is not None:
            img,label = self.transform(img,lbl = label)
        if self.target_transform:
            label = self.target_transform(label)

        return img,label
        
    def __len__(self) -> int:
        return len(self.list_samples)
