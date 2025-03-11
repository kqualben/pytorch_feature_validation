import os
from src.utils import open_pickle, plot_gradient_norms


model = 2503051229
model_dir = f"{os.getcwd()}/model_store/{model}"

grads = open_pickle(f"{model_dir}/gradient_norms.pkl")
print(len(grads))

plot = plot_gradient_norms(grads, len(grads))
plot.show()
