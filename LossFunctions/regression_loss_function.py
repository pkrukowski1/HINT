"""
    This file implements regression loss function used in the framework.
"""

import torch
import torch.nn as nn

class IntervalMSELoss(nn.Module):
    """
        The class implements an interval-based version of the Mean Square Error (MSE) loss. 
        Let us denote X = [x_0-eps, x_0+eps] as a hyperrectangle, where x_0 is the midpoint, and eps represents the radii.
        Let Y be the ground truth (GT) value calculated for the point x_0. The interval MSE is then defined as:
              IntervalMSE(X,Y) := 1/N * ||Y - X||_2^2
                           = 1/N * max{ ||Y-(x_0-eps)||_2^2, ||Y-(x_0+eps)||_2^2 },
        where N denotes number of points.
    """

    def __init__(self):
        super().__init__()

    def forward(self, lower_bound: torch.Tensor, 
                upper_bound: torch.Tensor, 
                y_values: torch.Tensor):
        
        """
        Parameters:
        -----------
            lower_bound: torch.Tensor
                Lower bound of a target network output.
            upper_bound: torch.Tensor
                Upper bound of a target network output.
            y_values: torch.Tensor
                GT values.
        
        Returns:
        --------
            Interval MSE loss.
        """
        
        lower_bound_loss = (y_values - lower_bound).pow(2).sum(dim=-1)
        upper_bound_loss = (y_values - upper_bound).pow(2).sum(dim=-1)
        max_loss = torch.maximum(lower_bound_loss, upper_bound_loss)

        return max_loss.mean()
