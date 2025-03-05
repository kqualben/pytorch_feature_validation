import numpy as np
import shap
import os
import torch
from src.data_processing import Preproceessing
from torch.utils.data import DataLoader
from src.model import ClassifierModel
from src.configs import ModelConfig
from src.utils import open_json
import matplotlib.pyplot as plt

model_id = 2502221426
model_path = f"{os.getcwd()}/model_store/{model_id}"
train_config = open_json(f"{model_path}/train_config.json")

model_config = ModelConfig.from_dict(**train_config)

data_proc = Preproceessing(root=os.getcwd())
model = ClassifierModel(input_n=model_config.input_n, hidden_n=model_config.hidden_n, num_classes=model_config.num_classes)
model.load_state_dict(torch.load(f"{model_path}/model_state.pt", weights_only=True))

train_loader = DataLoader(data_proc.train_set, batch_size=model_config.batches, shuffle=True)

background = next(iter(train_loader))[0][:1000] #[0] gets the images, [1] are the labels
shap_images = next(iter(train_loader))[0][:100] 

e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(shap_images)
print(shap_values.shape[0])
#sum values to get a single value per pixel
shap_values_class_sum = np.sum(shap_values, axis=-1)
shap_values_flat = shap_values_class_sum.reshape(shap_values.shape[0], -1) #flatten, shap_values.shape[0] == num images
test_images_flat = shap_images.view(shap_values.shape[0], -1).cpu().numpy()

fig = shap.summary_plot(shap_values_flat, test_images_flat, feature_names=[f'pixel_{i}' for i in range(shap_values_flat.shape[1])],show=False)
plt.savefig(f"{model_path}/shap_summary_plot.png", bbox_inches='tight')
plt.close()
