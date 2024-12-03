"""
This file implements worst-case loss function used in the framework.
"""

import torch
import torch.nn as nn

class IBP_Loss(nn.Module):
    """
    Custom loss function for Interval Bound Propagation (IBP).

    Attributes:
    -----------
    bce_loss_func: nn.CrossEntropyLoss
        Cross-entropy loss function.
    worst_case_error: float
        Worst-case prediction error.

    Properties:
    -----------
    worst_case_error: float
        Getter for the worst-case prediction error.

    Methods:
    --------
    forward(y_pred, y, z_l, z_u, kappa=0.5):
        Calculates the total loss for IBP.

    Returns:
    --------
    total_loss: torch.Tensor
        Total calculated loss.
    """
    def __init__(self):
        super().__init__()
        self.bce_loss_func = nn.CrossEntropyLoss()
        self._worst_case_error = 0.0

    @property
    def worst_case_error(self):
        """
        Getter for the worst-case prediction error.
        """
        return self._worst_case_error
    
    @worst_case_error.setter
    def worst_case_error(self, value):
        """
        Setter for the worst-case prediction error.
        """
        self._worst_case_error = value

    def forward(self, y_pred, y, z_l, z_u, *args, **kwargs):
        """
        Calculates the total loss for IBP.

        Args:
        -----
        y_pred: torch.Tensor
            Predicted logits.
        y: torch.Tensor
            Ground-truth labels.
        z_l: torch.Tensor
            Tensor with lower logits.
        z_u: torch.Tensor
            Tensor with upper logits.
        kappa: float, optional
            Weighting factor for combining fit loss and worst-case loss.

        Returns:
        --------
        total_loss: torch.Tensor
            Total calculated loss.
        """

        kappa = kwargs["kappa"]

        # Standard cross-entropy loss component
        loss_fit = self.bce_loss_func(y_pred, y)

        # Worst-case loss component
        tmp = nn.functional.one_hot(y, y_pred.size(-1))
        
        # Calculate worst-case prediction logits
        z = torch.where(tmp.bool(), z_l, z_u)

        # Calculate worst-case component error
        loss_spec = self.bce_loss_func(z, y)

        self.worst_case_error = (z.argmax(dim=1) != y).float().sum().item()
       
        # Calculate total loss
        total_loss = kappa * loss_fit + (1 - kappa) * loss_spec

        return total_loss
