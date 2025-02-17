from dataclasses import dataclass
from typing import Optional
from torch.utils.data import Dataset


@dataclass
class DataLoaderConfig:
    dataset: Dataset
    batch : int
    test_size : float = 0.7 
    shuffle : Optional[bool] = None
    drop_last: Optional[bool] = None

@dataclass
class TrainConfig:
    epochs: int
    batches: int
    learning_rate: float
    save_fig: bool
        






