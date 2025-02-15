from dataclasses import dataclass
from typing import Type, Optional
from torch.utils.data import Dataset


@dataclass
class DataLoaderConfig:
    dataset: Dataset
    batch : int
    test_size : float = 0.7 
    shuffle : Optional[bool] = None
    drop_last: Optional[bool] = None
        






