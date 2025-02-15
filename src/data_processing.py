from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2

from torchvision.datasets import MNIST

RANDOM_SEED = 30
TRAIN_TRANSFORMS = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
   

class Preproceessing:
    def __init__(self, root: Optional[str]):
        self.root = root
        self._train_set = MNIST(root = self.root, train=True, transform=TRAIN_TRANSFORMS)
        self.classes = self._train_set.classes
        #self._train_set, self._val_set = self.split_train_set(train_data)
        self._test_set = MNIST(root = self.root, train=False)
        
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
    
    @staticmethod
    def split_train_set(dataset: Dataset, train_size: float = 0.7):
        split_n = int(train_size * len(dataset))
        train, val = random_split(
            dataset,
            [split_n, len(dataset) - split_n],
            generator=torch.Generator().manual_seed(RANDOM_SEED),
        )
        return train, val
    
    @property
    def train_set(self):
        return self._train_set
    
    @property
    def val_set(self):
        return self._val_set
    
    @property
    def test_set(self):
        return self._test_set
    
    @property
    def num_classes(self):
        return len(self.classes)
    
    @property
    def train_loader(self):
        if self._train_loader is None:
            self._train_loader = DataLoader(self.train_set, shuffle=True,drop_last=True)
        return self._train_loader
    
    @property
    def test_loader(self):
        if self._test_loader is None:
            self._test_loader = DataLoader(self.test_set, shuffle=False, drop_last=False)
        return self._test_loader

