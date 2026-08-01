import torch
import torch.nn as nn
from sklearn.datasets import make_moons

# Dataset
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 32)   # input: 2 numbers (x, y coords) -> 8 hidden features
        self.layer2 = nn.Linear(32, 1)   # 32 hidden features -> 1 output (probability of class 1)
        self.relu = nn.ReLU() # non-linearity
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      x = self.layer1(x)   # <- pass through the first linear layer
      x = self.relu(x)
      x = self.layer2(x)   # <- pass through the second linear layer
      x = self.sigmoid(x)
      return x

model = SimpleNet()

criterion = nn.BCELoss() # binary cross-entropy (for binary class problem)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) # adjust weight using optimizer to reduce error

# Training
for epoch in range(2000):
    outputs = model(X)
    loss = criterion(outputs, y)

    # clear > calculate loss > apply the fix > go backward > restart process

    optimizer.zero_grad() # clear old gradients; make 0 updates to that neuron's weights
    loss.backward() # compute how much each weight contributed to the error (backpropagation)
    optimizer.step() # actually update the weights using those gradients

    if epoch % 100 == 0: # prints after every 40 epochs
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
