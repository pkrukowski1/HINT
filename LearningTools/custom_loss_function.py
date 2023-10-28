"""
This file implements a class with custom loss function used in an interval bound propagation neural networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IBP_Loss(nn.Module):

    def __init__(self, calculate_area_mode=False):
        super().__init__()
        self.bce_loss_func  = nn.CrossEntropyLoss()
        self.calculate_area_mode = calculate_area_mode
    
    def forward(self, mu_pred, y, eps_pred, eps, kappa=0.5):
        """
        Arguments:
        ----------

        mu_pred (torch.Tensor): tensor with logits corresponding to correct classification of data point
        y (torch.Tensor): ground-truth labels
        eps_pred (torch.Tensor): tensor with predicted radii
        eps (torch.Tensor): tensor with perturbated epsilon
        kappa (float): coefficient which is a trade-off between discriminant border and balls drawed
                       in the $L^{\infty}$ metric
        """

        # calculate lower and upper logits
        z_l = mu_pred - eps_pred
        z_u = mu_pred + eps_pred

        # standard cross-entropy loss component
        loss_fit = self.bce_loss_func(mu_pred, y)

        # worst-case loss component
        tmp = nn.functional.one_hot(y, mu_pred.size(-1))
        z = torch.where(tmp.bool(), z_l, z_u)

        loss_spec = self.bce_loss_func(z,y)

        # MSE loss corresponding to lengths of radii
        loss_eps = (eps_pred.sum(dim=1).mean() - eps.sum(dim=1).mean()).pow(2) if not self.calculate_area_mode \
              else (eps_pred.prod(dim=1).mean() - eps.prod(dim=1).mean()).pow(2)

        # total loss calculation
        total_loss = kappa * loss_fit + (1-kappa) * loss_spec + loss_eps
        
        return total_loss
