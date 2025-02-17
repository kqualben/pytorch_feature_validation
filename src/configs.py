from dataclasses import dataclass, fields
from typing import Type


@dataclass
class TrainConfig:
    epochs: int
    batches: int
    learning_rate: float
    save_fig: bool

@dataclass
class ModelConfig:
    epochs: int
    batches: int
    learning_rate: float
    num_classes: int
    input_n: int
    hidden_n: int

    @classmethod
    def from_dict(cls, **kwargs):
        # valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in {f.name for f in fields(cls)}}
        return cls(**filtered)
    