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

    @property
    def worst_case_error(self):
        """
        Getter for the worst-case MSE.
        """
        return self._worst_case_error
    
    @worst_case_error.setter
    def worst_case_error(self, value):
        """
        Setter for the worst-case MSE.
        """
        self._worst_case_error = value

    def forward(self, z_l: torch.Tensor, z_u: torch.Tensor, y: torch.Tensor, *args, **kwargs):
        
        """
        Parameters:
        -----------
            z_l: torch.Tensor
                Lower bound of a target network output.
            z_u: torch.Tensor
                Upper bound of a target network output.
            y: torch.Tensor
                GT values.
        
        Returns:
        --------
            Interval MSE loss.
        """
        
        lower_bound_loss = (y - z_l).pow(2).sum(dim=-1)
        upper_bound_loss = (y - z_u).pow(2).sum(dim=-1)
        max_loss = torch.maximum(lower_bound_loss, upper_bound_loss)
        loss = max_loss.mean()

        self.worst_case_error = loss.item()

        return loss
