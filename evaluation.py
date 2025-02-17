from src.utils import open_pickle, plot_losses

model = 2502171144
train, test = open_pickle(f"/Users/kristina.qualben/Desktop/pytorch_feature_validation/model_store/{model}/train_test_losses.pkl")

plot_losses(train, test, save=False)
