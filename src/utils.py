from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import json
import os
import pickle
import logging


def plot_losses(train: List[float], test: List[float], save: Optional[str]) -> plt.figure:
    """
    function to plot train vs test losses.

    :param List[float] train: list of train losses
    :param List[float] test: list of test losses
    :param Optional[str] save: save plot to given path. If falsey, then the plot wont be saved.

    :return plt.figure:
    """
    plt.figure()
    plt.plot(train, label='Train Loss', color='blue')
    plt.plot(test, label= 'Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save)
        plt.close()
    return plt.show()

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
