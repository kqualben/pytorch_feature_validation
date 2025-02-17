import os

from src.configs import TrainConfig
from src.model import Trainer

train_dict = {
    "epochs": 10,
    "batches": 32,
    "learning_rate": 0.0001,
    "save_fig": True,
}

train_config = TrainConfig(**train_dict)

trainer = Trainer(os.getcwd(), train_config)
trainer.train_eval()
