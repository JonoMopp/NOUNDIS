import numpy as np
import torch

class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.numpy()
