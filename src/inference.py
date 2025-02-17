import os
from model import Trainer
import matplotlib.pyplot as plt

trainer = Trainer(os.getcwd())
train_losses, test_losses = trainer.train(10)

plt.figure()
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(test_losses, label= 'Test Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Test Loss')
plt.legend()
plt.show()


trainer.predict()