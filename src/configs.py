from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int
    batches: int
    learning_rate: float
    save_fig: bool
