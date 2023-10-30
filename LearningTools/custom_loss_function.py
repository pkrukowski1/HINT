"""
This file implements a class with custom loss function used in an interval bound propagation neural networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IBP_Loss(nn.Module):

    def __init__(self, calculation_area_mode=False):
        super().__init__()
        self.bce_loss_func         = nn.CrossEntropyLoss()
        self.calculation_area_mode = calculation_area_mode
    
    def forward(self, y_pred, y, z_l, z_u, kappa=0.5):
        """
        Arguments:
        ----------

        mu_pred (torch.Tensor): tensor with logits corresponding to correct classification of data point
        y (torch.Tensor): ground-truth labels
        mu_pred (torch.Tensor): tensor with middles of intervals
        z_l (torch.Tensor): tensor with lower logits
        z_u (torch.Tensor): tensor with upper logits
        kappa (float): coefficient which is a trade-off between discriminant border and balls drawed
                       in the $L^{\infty}$ metric
        """

        # standard cross-entropy loss component
        loss_fit = self.bce_loss_func(y_pred, y)

        # worst-case loss component
        tmp = nn.functional.one_hot(y, y_pred.size(-1))
        
        z = torch.where(tmp.bool(), z_l, z_u)

        loss_spec = self.bce_loss_func(z,y)
        # # MSE loss corresponding to lengths of radii
        # loss_eps = (eps_pred.sum(dim=1).mean() - eps.sum(dim=1).mean()).pow(2) if not self.calculation_area_mode \
        #       else (eps_pred.prod(dim=1).mean() - eps.prod(dim=1).mean()).pow(2)
        
        # total loss calculation
        total_loss = kappa * loss_fit + (1-kappa) * loss_spec #+ loss_eps
        return total_loss
