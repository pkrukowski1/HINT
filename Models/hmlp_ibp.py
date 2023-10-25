"""
This file implements neccessary logic for applying interval bound propagation
over tasks' embeddings being inputs to an MLP hypernetwork
"""

from hypnettorch.hnets import HMLP
from hypnettorch.hnets.hnet_interface import HyperNetInterface

import torch
import torch.nn as nn

class HMLP_IBP(HMLP, HyperNetInterface):

    """
    Implementation of a `full hypernet` with interval bound propagation mechanism around tasks' embeddings.

    The network will consist of several hidden layers and a final linear output
    layer that produces all weight matrices/bias-vectors the network has to
    produce.

    The network allows to maintain a set of embeddings internally that can be
    used as conditional input.
    """

    def __init__(self, eps_perturbated=0.05): 
        super().__init__(activation_fn=nn.ReLU()) # for now only ReLU is supported
        self.eps_perturbated = eps_perturbated
    
    def forward(self, uncond_input=None, cond_input=None, cond_id=None,
                weights=None, distilled_params=None, condition=None,
                ret_format='squeezed'):
        """Compute the weights of a target network.

        Args:
            (....): See docstring of method
                :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
            condition (int, optional): This argument will be passed as argument
                ``stats_id`` to the method
                :meth:`utils.batchnorm_layer.BatchNormLayer.forward` if batch
                normalization is used.

        Returns:
            (list or torch.Tensor): See docstring of method
            :meth:`hnets.hnet_interface.HyperNetInterface.forward`.
        """
        uncond_input, cond_input, uncond_weights, _ = \
            self._preprocess_forward_args(uncond_input=uncond_input,
                cond_input=cond_input, cond_id=cond_id, weights=weights,
                distilled_params=distilled_params, condition=condition,
                ret_format=ret_format)

        ### Prepare hypernet input ###
        assert self._uncond_in_size == 0 or uncond_input is not None
        assert self._cond_in_size == 0 or cond_input is not None
        if uncond_input is not None:
            assert len(uncond_input.shape) == 2 and \
                   uncond_input.shape[1] == self._uncond_in_size
            h = uncond_input
        if cond_input is not None:
            assert len(cond_input.shape) == 2 and \
                   cond_input.shape[1] == self._cond_in_size
            h = cond_input
        if uncond_input is not None and cond_input is not None:
            h = torch.cat([uncond_input, cond_input], dim=1)

        ### Extract layer weights ###
        bn_scales = []
        bn_shifts = []
        fc_weights = []
        fc_biases = []

        assert len(uncond_weights) == len(self.unconditional_param_shapes_ref)
        for i, idx in enumerate(self.unconditional_param_shapes_ref):
            meta = self.param_shapes_meta[idx]

            if meta['name'] == 'bn_scale':
                bn_scales.append(uncond_weights[i])
            elif meta['name'] == 'bn_shift':
                bn_shifts.append(uncond_weights[i])
            elif meta['name'] == 'weight':
                fc_weights.append(uncond_weights[i])
            else:
                assert meta['name'] == 'bias'
                fc_biases.append(uncond_weights[i])

        if not self.has_bias:
            assert len(fc_biases) == 0
            fc_biases = [None] * len(fc_weights)

        if self._use_batch_norm:
            assert len(bn_scales) == len(fc_weights) - 1

        ### Process inputs through network ###
        eps_per_T = self.eps_perturbated * torch.ones_like(h) # perturbation radii

        # for i in range(0, len(self._hidden_dims), 2 if self._use_bias else 1):
        #     b = None
        #     if self._use_bias:
        #         b = weights[i+1]

        #     per_eps = per_eps @ torch.abs(weights[i]).T
        #     h = h @ weights[i].T + b[i]

        #     if self._act_fn is not None:
        #         # Right now this method works only with ReLU function
        #         # h = self._act_fn(h)

        #         z_l, z_u = h - per_eps, h + per_eps
        #         z_l, z_u = F.relu(z_l), F.relu(z_u)
        #         h, per_eps = (z_u + z_l) / 2, (z_u - z_l) / 2

        #     if self._dropout is not None:
        #         h = self._dropout(h)
        for i in range(len(fc_weights)):
            eps_per_T = eps_per_T @ torch.abs(fc_weights[i]).T # Update radii
            h = h @ fc_weights[i].T + fc_biases[i]             # Update embeddings

            # Apply non-linearity
            h = self._act_fn(h)

            # Calculate lower and upper logit
            z_l, z_u = h - eps_per_T, h + eps_per_T
            z_l, z_u = self._act_fn(z_l), self._act_fn(z_u)            

        ### Split output into target shapes ###
        ret = self._flat_to_ret_format(h, ret_format)

        return ret, z_l, z_u, eps_per_T