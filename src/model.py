from src.data_processing import Preproceessing
from src.configs import TrainConfig
from src.utils import plot_losses, save_json, save_pickle, logger
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Tuple, List
from torchmetrics import Precision, Recall
from torch.utils.data import DataLoader
import datetime
import os
from dataclasses import asdict


class ClassifierModel(nn.Module):
  def __init__(self, input_n: int, hidden_n: int, num_classes: int):
    super(ClassifierModel, self).__init__()
    self.flatten = nn.Flatten() 
    self.model = nn.Sequential(
        nn.Linear(input_n, hidden_n),
        nn.BatchNorm1d(hidden_n),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_n, hidden_n //2),
        nn.BatchNorm1d(hidden_n // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_n // 2, num_classes)
    )

  def forward(self, x):
    if len(x.shape) > 2:
        x = self.flatten(x)
    return self.model(x)

class Trainer(Preproceessing):
    def __init__(self, root:str, train_config: TrainConfig):
        super().__init__(root)
        self.training_config = train_config
        self.base_path = f"{root}/model_store"
        self.batch_n = self.training_config.batches
        self.learning_rate = self.training_config.learning_rate
        self.input_n = 784
        self.train_loader = DataLoader(self.train_set, batch_size = self.batch_n, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size = self.batch_n, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size = self.batch_n, shuffle=False)
        self.model = ClassifierModel(input_n=self.input_n, hidden_n=128, num_classes=self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr = self.learning_rate)
        
    def train_epoch(self) -> int:
        self.model.train()
        train_loss = 0
        correct, total = 0, 0
        for data, label in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(output.data, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
        epoch_loss = train_loss / len(self.train_loader)
        self.logger.info(f">> Training Loss: {epoch_loss:.2f}, Accuracy: {((correct/total)):.2f}")
        return epoch_loss

    def val_loss(self) -> int:
        self.model.eval()
        total_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for data, label in self.val_loader:
                if len(label.shape) > 1:
                    label = label.squeeze()
                output = self.model(data)
                loss = self.criterion(output, label)
                total_loss += loss.item()
                _, pred = torch.max(output.data, 1)
                total += label.size(0)
                correct += (pred == label).sum().item()
        eval_loss = total_loss / len(self.val_loader)
        self.logger.info(f'>> Validate Loss: {eval_loss:.2f} Accuracy: {(correct / total):.2f}')
        return eval_loss


    def train(self, num_epochs:int) -> Tuple[List, List]:
        train_losses = []
        test_losses = []
        for e in range(num_epochs):
            self.logger.info(f"\nEpoch: {e}")
            epoch_loss = self.train_epoch()
            train_losses.append(epoch_loss)

            test_loss = self.val_loss()
            test_losses.append(test_loss)

        return train_losses, test_losses

    def eval(self) -> None:
        self.model.eval()
        correct,  total = 0, 0
        metric_precision = Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        metric_recall = Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        with torch.no_grad():
            for data, label in self.test_loader:
                output = self.model(data)
                _, pred = torch.max(output.data, 1)
                total += label.size(0)
                correct += (pred == label).sum().item()
                metric_precision(pred, label)
                metric_recall(pred, label)
            acc = correct / total
            precision = metric_precision.compute().item()
            recall = metric_recall.compute().item()
        self.logger.info(f'Prediction Accuracy: {100 * acc:.2f}%')
        self.logger.info(f'Precision: {precision:.2f}')
        self.logger.info(f'Recall: {recall:.2f}')
        return acc, precision, recall
    
    def train_eval(self) -> None:
        epochs = self.training_config.epochs
        save_fig = self.training_config.save_fig
        #Train
        model_dir = f"{self.base_path}/{datetime.datetime.today().strftime('%y%m%d%H%M')}"
        os.mkdir(model_dir)
        self.logger = logger(
            directory=model_dir,
            filename=f"training_log.log",
        )
        train_losses, test_losses = self.train(epochs)
        torch.save(
            self.model.state_dict(),
            f"{model_dir}/model_state.pt",
        )
        save_pickle([train_losses, test_losses], model_dir, "train_test_losses.pkl")
        if save_fig:
            save_fig = f"{model_dir}/train_loss_plot.png"
        plot_losses(train_losses, test_losses, save=save_fig)
        #Final Evaluation
        accuracy, precision, recall = self.eval()
        log_dict = asdict(self.training_config)
        log_dict.update({"accuracy": accuracy, "precision": precision, "recall" : recall})
        save_json(log_dict, model_dir, "train_config.json" )
