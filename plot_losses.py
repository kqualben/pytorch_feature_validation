import os
from src.utils import open_pickle, plot_losses


model = 2502191428
model_dir = f"{os.getcwd()}/model_store/{model}"

train, test = open_pickle(f"{model_dir}/train_test_losses.pkl")
plot = plot_losses(train, test)
plot.show()
