import os
from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr
from typing import List

class_eval = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 15, 14, 12, 9, 10]

##############################################################################################################################################################################

class IDDADataset(VisionDataset):

    def __init__(self,
                 root: str,
                 list_samples: List[str],
                 transform: tr.Compose = None,
                 client_name: str = None):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.client_name = client_name
        self.target_transform = self.get_mapping()
        self.return_PIL = False

    @staticmethod
    def get_mapping():
        classes = class_eval
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for i, cl in enumerate(classes):
            mapping[i] = cl
        return lambda x: from_numpy(mapping[x])

    def __getitem__(self, index: int) -> Any:

        sample_name = self.list_samples[index]
        img_path = os.path.join('data/IDDAv3/idda/images', sample_name) + ".jpg"
        img = Image.open(img_path).convert('RGB')
        label_path = os.path.join('data/IDDAv3/idda/labels', sample_name) + '.png'
        label = Image.open(label_path).convert('L')
        
        if self.transform is not None and not self.return_PIL:
            img,label = self.transform(img,lbl = label)        
        if self.target_transform and not self.return_PIL:
            label = self.target_transform(label)
            
        return img,label
        
    def __len__(self) -> int:
        return len(self.list_samples)
