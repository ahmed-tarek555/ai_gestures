import torch
from utils import load_dataset, get_batch, model_path
from model import Model

n_iter = 1000
lr = 1e-5

model  = Model()
inputs, targets, classes = load_dataset("hands")

optim = torch.optim.AdamW(model.parameters(), lr)
model.train()
for _ in range(n_iter):
    optim.zero_grad()
    x, y = get_batch(inputs, targets)
    loss = model(x, y)
    print(f"Loss: {loss.item()}")
    loss.backward()
    optim.step()

torch.save(model.state_dict(), model_path)
print(f'Parameters saved to {model_path}')

