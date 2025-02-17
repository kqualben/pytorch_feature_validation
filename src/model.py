from data_processing import Preproceessing
import torch.nn as nn
from torch.optim import Adam
import torch
from typing import Tuple, List,Optional
from torchmetrics import Precision, Recall
from torch.utils.data import DataLoader

LEARNING_RATE = 0.001

class ClassifierModel(nn.Module):
  def __init__(self, input_n: int, hidden_n: int, num_classes: int):
    super(ClassifierModel, self).__init__()
    self.flatten = nn.Flatten() 
    self.model = nn.Sequential(
        nn.Linear(input_n, hidden_n),
        nn.ReLU(),
        nn.Linear(hidden_n, hidden_n),
        nn.ReLU(),
        nn.Linear(hidden_n, num_classes),
    )

  def forward(self, x):
    if len(x.shape) > 2:
        x = self.flatten(x)
    return self.model(x)

class Trainer(Preproceessing):
    def __init__(self, root:str, batch_n: Optional[int] = 32):
        super().__init__(root)
        self.batch_n = batch_n
        self.input_n = 784
        self.train_loader = DataLoader(self.train_set, batch_size = self.batch_n, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size = self.batch_n, shuffle=False)
        self.model = ClassifierModel(input_n=self.input_n, hidden_n=128, num_classes=self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr = LEARNING_RATE)
        
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
            print(f">> Loss: {epoch_loss:.2f}, Accuracy: {(100*(correct/total))}")
            return epoch_loss

    def eval_loss(self) -> int:
        self.model.eval()
        total_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for data, label in self.test_loader:
                if len(label.shape) > 1:
                    label = label.squeeze()
                output = self.model(data)
                loss = self.criterion(output, label)
                total_loss += loss.item()
                _, pred = torch.max(output.data, 1)
                total += label.size(0)
                correct += (pred == label).sum().item()
        print(f'>> Test Accuracy: {100 * correct / total:.2f}%')
        return total_loss / len(self.test_loader)


    def train(self, num_epochs:int) -> Tuple[List, List]:
        train_losses = []
        test_losses = []
        for e in range(num_epochs):
            print(f"\nEpoch: {e}")
            epoch_loss = self.train_epoch()
            train_losses.append(epoch_loss)

            test_loss = self.eval_loss()
            test_losses.append(test_loss)

        return train_losses, test_losses

    def predict(self) -> None:
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
            print(f'Prediction Accuracy: {100 * correct / total:.2f}%')
            print(f'Precision: {metric_precision.compute().item():.2f}')
            print(f'Recall: {metric_recall.compute().item():.2f}')
        return None
    