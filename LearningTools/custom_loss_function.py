"""
This file implements a class with custom loss function used in an interval bound propagation neural networks
"""

import torch
import torch.nn as nn

class IBP_Loss(nn.Module):

    def __init__(self, calculation_area_mode=False):
        super().__init__()
        self.bce_loss_func         = nn.CrossEntropyLoss()
        self.calculation_area_mode = calculation_area_mode
    
    def forward(self, y_pred, y, z_l, z_u, kappa=0.5):
        """
        Arguments:
        ----------

        y (torch.Tensor): ground-truth labels
        z_l (torch.Tensor): tensor with lower logits
        z_u (torch.Tensor): tensor with upper logits
        kappa (float): coefficient which is a trade-off between discriminant border and balls drawed
                       in the $L^{\infty}$ metric

        Returns:
        --------

        total_loss (torch.Tensor): total calculated loss
        """

        # standard cross-entropy loss component
        loss_fit = self.bce_loss_func(y_pred, y)

        # worst-case loss component
        tmp = nn.functional.one_hot(y, y_pred.size(-1))
        
        # calculate worst-case prediction logits
        z = torch.where(tmp.bool(), z_l, z_u)

        loss_spec = self.bce_loss_func(z,y)
        
        total_loss = kappa * loss_fit + (1-kappa) * loss_spec

        return total_loss
