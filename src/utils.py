import json
import logging
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns


def plot_losses(train: List[float], test: List[float]) -> plt.figure:
    """
    function to plot train vs test losses.

    :param List[float] train: list of train losses
    :param List[float] test: list of test losses

    :return plt.figure:
    """
    plt.figure()
    plt.plot(train, label="Train Loss", color="blue")
    plt.plot(test, label="Test Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Test Loss")
    plt.legend()
    plt.tight_layout()
    return plt

def plot_gradient_norms(gradient_norms: List[float], num_epochs: int) -> plt.figure:
    """
    function to plot gradient norms.

    :param List[float] gradient_norms: list of gradients
    :param float num_epochs: number of epochs used in training

    :return plt.figure:
    """
    plt.figure()
    plt.plot(range(num_epochs), gradient_norms)
    plt.xlabel("Epochs")
    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norms Over Epochs")
    plt.yscale("log")
    plt.tight_layout()
    return plt


def plot_correlation_matrix(corr_matrix, layer_name: str) -> plt.figure:
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix.numpy(), cmap="coolwarm", center=0)
    plt.title(f"Feature Correlations - {layer_name}")
    plt.tight_layout()
    return plt


def plot_activations_distributions(activations: Dict) -> plt.figure:
    fig, axes = plt.subplots(len(activations), 2, figsize=(15, 5 * len(activations)))

    for idx, (name, acts) in enumerate(activations.items()):
        acts_flat = acts.reshape(-1).numpy()

        # Histogram
        axes[idx, 0].hist(acts_flat, bins=50)
        axes[idx, 0].set_title(f"{name} - Distribution")
        axes[idx, 0].set_xlabel("Activation Value")
        axes[idx, 0].set_ylabel("Frequency")

        # Box plot
        axes[idx, 1].boxplot(acts_flat)
        axes[idx, 1].set_title(f"{name} - Box Plot")

    plt.tight_layout()
    return plt


def save_json(data: dict, directory: str, filename: str) -> str:
    """
    function to save data dictionary to given directory/filename.

    :param dict data: dictionary
    :param str directory: root directory where file will be saved.
    :param str filename: name and file type.

    :return str: path to where file has been saved.
    """
    location = os.path.join(directory, filename)
    with open(location, "w") as f:
        json.dump(data, f)
    print(f"File saved to: {location}")
    return location


def open_json(path: str) -> Dict:
    """
    function to open an json file given it's path.

    :param str path: path to json file.

    :return dict:
    """
    print(f"Loading: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_pickle(data, directory: str, filename: str) -> str:
    """
    function to save data object to given directory/filename.

    :param pytorch.object data: pytorch model object
    :param str directory: root directory where file will be saved.
    :param str filename: name and file type.

    :return str: path to where file has been saved.
    """
    location = os.path.join(directory, filename)
    with open(location, "wb") as f:
        pickle.dump(data, f)
    print(f"File saved to: {location}")
    return location


def open_pickle(path: str) -> Dict:
    """
    function to open an pickle file given it's path.

    :param str path: path to pickle file.

    :return dict:
    """
    print(f"Loading: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def logger(directory: str, filename: str):
    """
    function to spin up logger module which gets saved to given directory/filename.

    :param str directory: root directory where file will be saved
    :param str filename: name and file type

    :return logging.object:
    """
    logging.basicConfig(
        filename=f"{directory}/{filename}",
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger
