import torch.nn as nn
import torch.nn.functional as F
from utils import input_size, classes

n_hidden = 512
n_classes = len(classes)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden//2)
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_hidden//2, n_classes)
        )

    def forward(self, x, y=None):
        x = self.fc(x)
        embeddings = x
        logits = self.classifier(x)
        if y is not None:
            loss = F.cross_entropy(logits, y)
            return loss
        else:
            return F.normalize(embeddings, dim=0)
