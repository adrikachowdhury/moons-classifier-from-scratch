import numpy as np
import torch
import matplotlib.pyplot as plt
from train import model, X, y  # reuses the trained model + data from train.py

"""
np.meshgrid, .ravel(), np.c_[...], .reshape(xx.shape)
- build and reshape a grid of coordinates
"""

# create a grid covering the data's range
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# grid- building a large set of test points covering 2D space
# torch.tensor- just predicting, not training
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# asking for predictions, not any learning
with torch.no_grad():
    preds = model(grid).reshape(xx.shape) # model's preds at every point on that grid

#contourf- draws preds as colored bg
# scatter- draws real data points on top
plt.contourf(xx, yy, preds, levels=50, cmap="RdBu", alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap="RdBu", edgecolors="k")
plt.title("What the network learned")
plt.show()
