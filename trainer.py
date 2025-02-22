import os

from src.configs import TrainConfig
from src.model import Trainer

train_dict = {
    "epochs": 15,
    "batches": 32,
    "learning_rate": 0.0001,
}

train_config = TrainConfig(**train_dict)

trainer = Trainer(os.getcwd(), train_config)
trainer.train_eval()
