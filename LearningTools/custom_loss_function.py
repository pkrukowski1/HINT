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
        self.worst_case_error      = 0.0
    

    # def forward(self, y_pred, y, z_l, z_u, radii, eps, kappa=0.5):
    def forward(self, y_pred, y, z_l, z_u, radii, eps, kappa=0.5):

        """
        Arguments:
        ----------

        y (torch.Tensor): ground-truth labels
        z_l (torch.Tensor): tensor with lower logits
        z_u (torch.Tensor): tensor with upper logits
        radii (torch.Tensor): tensor with predicted radii of intervals
        eps (torch.Tensor): tensor with perturbated epsilon
        kappa (float): coefficient which is a trade-off between discriminant border and balls drawed
                       in the $L^{\infty}$ metric

        Returns:
        --------

        total_loss (torch.Tensor): total calculated loss
        """

        # Standard cross-entropy loss component
        loss_fit = self.bce_loss_func(y_pred, y)

        # Worst-case loss component
        tmp = nn.functional.one_hot(y, y_pred.size(-1))
        
        # Calculate worst-case prediction logits
        z = torch.where(tmp.bool(), z_l, z_u)

        # Calculate worst case component error
        loss_spec = self.bce_loss_func(z,y)

        self.worst_case_error = (z.argmax(dim=1) != y).float().sum().item()

        # Calculate radii assocciated loss
        loss_eps = (radii - eps).pow(2).mean()
        # loss_eps = -radii.mean()
        
        # Calculate total loss
        total_loss = kappa * loss_fit + (1-kappa) * loss_spec + loss_eps

        return total_loss
