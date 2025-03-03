import datetime
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall

from src.configs import TrainConfig
from src.data_processing import Preproceessing
from src.utils import (
    logger,
    plot_activations_distributions,
    plot_correlation_matrix,
    plot_losses,
    save_json,
    save_pickle,
)


class ClassifierModel(nn.Module):
    def __init__(self, input_n: int, hidden_n: int, num_classes: int):
        super(ClassifierModel, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(input_n, hidden_n),
            nn.BatchNorm1d(hidden_n),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_n, hidden_n // 2),
            nn.BatchNorm1d(hidden_n // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_n // 2, num_classes),
        )

    def forward(self, x):
        if len(x.shape) > 2:
            x = self.flatten(x)
        return self.model(x)


class Trainer(Preproceessing):
    def __init__(self, root: str, train_config: TrainConfig):
        super().__init__(root)
        self.training_config = train_config
        self.base_path = f"{root}/model_store"
        self.batch_n = self.training_config.batches
        self.learning_rate = self.training_config.learning_rate
        self.input_n = 784
        self.hidden_n = 128
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_n, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=self.batch_n, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_n, shuffle=False
        )
        self.model = ClassifierModel(
            input_n=self.input_n, hidden_n=self.hidden_n, num_classes=self.num_classes
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        # Create Hooks
        self.activations = {}
        self.handles = []
        for name, layer in self.model.named_modules():
            self.handles.append(layer.register_forward_hook(self._create_hook(name)))

    def _create_hook(self, name):
        def hook(module, _, output):
            self.activations[name] = output.detach()

        return hook

    def analyze_layer(self, name, layer_type):
        activation = self.activations[name]

        if isinstance(layer_type, nn.Linear):
            # Linear layers: Check output distribution and weight usage
            self.logger.info(f"\nLinear Layer {name}:")
            self.logger.info(f"Output shape: {activation.shape}")
            self.logger.info(f"Mean activation: {activation.mean().item():.4f}")
            self.logger.info(f"Activation std: {activation.std().item():.4f}")

        elif isinstance(layer_type, nn.BatchNorm1d):
            # BatchNorm: Check if normalization is working properly
            self.logger.info(f"\nBatchNorm Layer {name}:")
            self.logger.info(
                f"Mean (should be close to 0): {activation.mean().item():.4f}"
            )
            self.logger.info(
                f"Std (should be close to 1): {activation.std().item():.4f}"
            )

        elif isinstance(layer_type, nn.ReLU):
            # ReLU: Check sparsity and dead neurons
            self.logger.info(f"\nReLU Layer {name}:")
            sparsity = (activation == 0).float().mean().item()
            self.logger.info(f"Sparsity (% of zeros): {sparsity * 100:.2f}%")

        elif isinstance(layer_type, nn.Dropout):
            # Dropout: Verify dropout rate
            self.logger.info(f"\nDropout Layer {name}:")
            zeros = (activation == 0).float().mean().item()
            self.logger.info(f"Actual dropout rate: {zeros * 100:.2f}%")

    def run_layer_analyzer(self):
        for name, layer in self.model.named_modules():
            if not isinstance(layer, nn.Sequential):
                self.analyze_layer(name, layer)

    def summarize_activation_stats(self, activations: Dict[str, torch.Tensor]):
        stats = {}
        for name, acts in activations.items():
            stats[name] = {
                "mean": acts.mean().item(),
                "std": acts.std().item(),
                "min": acts.min().item(),
                "max": acts.max().item(),
                "zeros_pct": (acts == 0).float().mean().item() * 100,
            }
        return stats

    def compute_feature_correlations(self, layer_name: str) -> torch.Tensor:
        activations = self.activations[layer_name]

        if len(activations.shape) > 2:
            print(f"reshaping for correlation: {layer_name}")
            activations = activations.reshape(activations.shape[0], -1)

        return torch.corrcoef(activations.T)

    def train_epoch(self, index: int) -> int:
        self.model.train()
        train_loss = 0
        correct, total = 0, 0
        for batch_idx, (data, label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, label)
            if (
                (index == 0 and batch_idx == 0)
                or (loss.item() > train_loss * 2)
                or (batch_idx % 100 == 0)
            ):
                self.logger.info(
                    f"\nAnalyzing layers at epoch {index}, batch {batch_idx}"
                )
                self.run_layer_analyzer()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(output.data, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
        epoch_loss = train_loss / len(self.train_loader)
        self.logger.info(
            f">> Training Loss: {epoch_loss:.2f}, Accuracy: {((correct/total)):.2f}"
        )
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
        self.logger.info(
            f">> Validate Loss: {eval_loss:.2f} Accuracy: {(correct / total):.2f}"
        )
        return eval_loss

    def train(self, num_epochs: int) -> Tuple[List, List]:
        train_losses = []
        test_losses = []
        for e in range(num_epochs):
            self.logger.info(f"\nEpoch: {e}")
            epoch_loss = self.train_epoch(e)
            train_losses.append(epoch_loss)

            test_loss = self.val_loss()
            test_losses.append(test_loss)

        return train_losses, test_losses

    def eval(self) -> None:
        self.model.eval()
        correct, total = 0, 0
        metric_precision = Precision(
            task="multiclass", num_classes=self.num_classes, average="macro"
        )
        metric_recall = Recall(
            task="multiclass", num_classes=self.num_classes, average="macro"
        )
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
        self.logger.info(f"Prediction Accuracy: {100 * acc:.2f}%")
        self.logger.info(f"Precision: {precision:.2f}")
        self.logger.info(f"Recall: {recall:.2f}")
        return acc, precision, recall

    def train_eval(self) -> None:
        epochs = self.training_config.epochs
        model_dir = (
            f"{self.base_path}/{datetime.datetime.today().strftime('%y%m%d%H%M')}"
        )
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

        plot = plot_losses(train_losses, test_losses)
        path = f"{model_dir}/train_loss_plot.png"
        plot.savefig(path)
        plot.close()
        self.logger.info(f"Saved: {path}")

        self.logger.info(f"\nFinal Layer Analysis:")
        self.run_layer_analyzer()

        save_pickle(self.activations, model_dir, "activations.pkl")
        activation_stats = self.summarize_activation_stats(self.activations)
        save_json(activation_stats, model_dir, "activation_stats.json")
        plot = plot_activations_distributions(self.activations)
        path = f"{model_dir}/activation_distribution_plot.png"
        plot.savefig(path)
        self.logger.info(f"Saved: {path}")

        corr_matrices = {}

        os.mkdir(f"{model_dir}/corr_matrices/")
        for name, layer in self.model.named_modules():
            if not isinstance(layer, nn.Sequential):
                self.logger.info(f"Computing Feature Correlations for: {name}")
                corr_matrix = self.compute_feature_correlations(name)
                corr_matrices[name] = corr_matrix
                plot = plot_correlation_matrix(corr_matrix, name)
                corr_path = f"{model_dir}/corr_matrices/{name}_heatmap.png"
                plot.savefig(corr_path)
                self.logger.info(f"Saved: {corr_path}")
        save_pickle(corr_matrices, model_dir, "corr_matrices.pkl")

        accuracy, precision, recall = self.eval()
        log_dict = asdict(self.training_config)
        log_dict.update(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "num_classes": self.num_classes,
                "input_n": self.input_n,
                "hidden_n": self.hidden_n,
            }
        )
        save_json(log_dict, model_dir, "train_config.json")
        self.logger.info(f"All results logged to: {model_dir}")
