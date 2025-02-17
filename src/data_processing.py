from typing import Optional
import torch
from torch.utils.data import Dataset, random_split
from torchvision.transforms import v2

from torchvision.datasets import MNIST

RANDOM_SEED = 30
TRANSFORMS = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.1307,), (0.3081,)) 
])
   

class Preproceessing:
    def __init__(self, root: Optional[str]):
        self.root = root
        self._train_set = MNIST(root = self.root, train=True, transform=TRANSFORMS)
        self._train_set, self._val_set = self.split_train_set(self._train_set)
        self.classes = self._train_set.dataset.classes
        self._test_set = MNIST(root = self.root, train=False, transform=TRANSFORMS)
    
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
    
