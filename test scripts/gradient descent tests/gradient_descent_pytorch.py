import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
from copy import copy

class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, weights):
        super().__init__()
        # make weights torch parameters
        self.weights = nn.Parameter(weights)
        # self.current = current     

    def motion_model(self, p, x, y):
        """
            Given the current parameters and the coordinates of a point, computes the compensated coordinates.
        """
        x1 = int((p[0]+p[2]*x+p[3]*y)/(p[6]*x+p[7]*y+1))
        y1 = int((p[1]+p[4]*x+p[5]*y)/(p[6]*x+p[7]*y+1))
        return (x1,y1)
    
    def compute_compensated_frame(self, pre_frame, parameters):
        """
            Computes I' given I and the current parameters.
        """
        compensated = np.zeros_like(pre_frame)
        for i in range(compensated.shape[0]):
            for j in range(compensated.shape[1]):
                (x1, y1) = self.motion_model(parameters, i, j)
                # sanitize limits
                x1 = max(0,x1)
                x1 = min(x1,compensated.shape[0])
                y1 = max(0,y1)
                y1 = min(y1,compensated.shape[1])
                compensated[x1][y1] = pre_frame[i][j]
        return compensated

    def forward(self, X):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-k * X) + b),
        """
        parameters = self.weights
        compensated_frame = self.compute_compensated_frame(X, parameters)
        return compensated_frame
    
def training_loop(model, optimizer, prev, curr, n=5):
    "Training loop for torch model."
    losses = []
    for i in range(n):
        print(f"iteration {i}")
        preds = model(prev)
        y_pred = torch.from_numpy((preds.flatten()).astype('float32'))
        y_pred.requires_grad = True
        y_targ = torch.from_numpy((curr.flatten()).astype('float32'))
        loss = F.mse_loss(y_pred, y_targ).sqrt()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)  
    return losses