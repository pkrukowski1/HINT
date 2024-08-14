# Modification of hypnettorch file
# (https://hypnettorch.readthedocs.io/en/latest/_modules/hypnettorch/mnets/mlp.html#MLP)
# licensed under the Apache License, Version 2.0
#
# HyperInterval needs an interval version of MLP.

import torch
import torch.nn as nn
import numpy as np
from warnings import warn

from hypnettorch.mnets.mnet_interface import MainNetInterface
from hypnettorch.utils.torch_utils import init_params
from hypnettorch.mnets import MLP

from IntervalNets.interval_modules import (IntervalDropout, 
                                            IntervalLinear)

class IntervalMLP(MLP, MainNetInterface):
    """
    Implementation of a Multi-Layer Perceptron (MLP) which works on intervals

    This is a simple fully-connected network, that receives input vector
    :math:`\mathbf{x}` and outputs a vector :math:`\mathbf{y}` of real values.

    The output mapping does not include a non-linearity by default, as we wanna
    map to the whole real line (but see argument ``out_fn``).

    Parameters:
    -----------
    n_in : int
        Number of inputs.
    n_out : int
        Number of outputs.
    hidden_layers : list or tuple
        A list of integers, each number denoting the size of a hidden layer.
    activation_fn : torch.nn.Module, optional
        The nonlinearity used in hidden layers. If ``None``, no nonlinearity will be applied.
    use_bias : bool, optional
        Whether layers may have bias terms.
    no_weights : bool, optional
        If set to ``True``, no trainable parameters will be constructed, i.e., weights are assumed to be produced ad-hoc by a hypernetwork and passed to the :meth:`forward` method.
    init_weights : optional
        This option is for convenience reasons. The option expects a list of parameter values that are used to initialize the network weights. As such, it provides a convenient way of initializing a network with a weight draw produced by the hypernetwork. Note, internal weights will be affected by this argument only.
    dropout_rate : float, optional
        If ``-1``, no dropout will be applied. Otherwise a number between 0 and 1 is expected, denoting the dropout rate of hidden layers.
    use_spectral_norm : bool, optional
        Use spectral normalization for training.
    use_batch_norm : bool, optional
        Whether batch normalization should be used. Will be applied before the activation function in all hidden layers.
    bn_track_stats : bool, optional
        If batch normalization is used, then this option determines whether running statistics are tracked in these layers or not. If ``False``, then batch statistics are utilized even during evaluation. If ``True``, then running stats are tracked. When using this network in a continual learning scenario with different tasks then the running statistics are expected to be maintained externally.
    distill_bn_stats : bool, optional
        If ``True``, then the shapes of the batchnorm statistics will be added to the attribute ``hyper_shapes_distilled`` and the current statistics will be returned by the method ``distillation_targets``.
    use_context_mod : bool, optional
        Add context-dependent modulation layers after the linear computation of each layer.
    context_mod_inputs : bool, optional
        Whether context-dependent modulation should also be applied to network inputs directly. I.e., assume :math:`\mathbf{x}` is the input to the network. Then the first network operation would be to modify the input via :math:`\mathbf{x} \cdot \mathbf{g} + \mathbf{s}` using context-dependent gain and shift parameters.
    no_last_layer_context_mod : bool, optional
        If ``True``, context-dependent modulation will not be applied to the output layer.
    context_mod_no_weights : bool, optional
        The weights of the context-mod layers are treated independently of the option ``no_weights``. This argument can be used to decide whether the context-mod parameters (gains and shifts) are maintained internally or externally.
    context_mod_post_activation : bool, optional
        Apply context-mod layers after the activation function in hidden layer rather than before, which is the default behavior.
    context_mod_gain_offset : bool, optional
        Activates option ``apply_gain_offset`` of class ``ContextModLayer`` for all context-mod layers that will be instantiated.
    context_mod_gain_softplus : bool, optional
        Activates option ``apply_gain_softplus`` of class ``ContextModLayer`` for all context-mod layers that will be instantiated.
    out_fn : optional
        If provided, this function will be applied to the output neurons of the network.
    verbose : bool, optional
        Whether to print information (e.g., the number of weights) during the construction of the network.
"""
    def __init__(self, n_in=1, n_out=1, hidden_layers=(10, 10),
                 activation_fn=torch.nn.ReLU(), use_bias=True, no_weights=False,
                 init_weights=None, dropout_rate=-1, use_spectral_norm=False,
                 use_batch_norm=False, bn_track_stats=True,
                 distill_bn_stats=False, use_context_mod=False,
                 context_mod_inputs=False, no_last_layer_context_mod=False,
                 context_mod_no_weights=False,
                 context_mod_post_activation=False,
                 context_mod_gain_offset=False, context_mod_gain_softplus=False,
                 out_fn=None, verbose=True):
        
        MainNetInterface.__init__(self)
        MLP.__init__(self, n_in=n_in, n_out=n_out, hidden_layers=hidden_layers,
                 activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights,
                 init_weights=init_weights, dropout_rate=dropout_rate, use_spectral_norm=use_spectral_norm,
                 use_batch_norm=use_batch_norm, bn_track_stats=bn_track_stats,
                 distill_bn_stats=distill_bn_stats, use_context_mod=use_context_mod,
                 context_mod_inputs=context_mod_inputs, no_last_layer_context_mod=no_last_layer_context_mod,
                 context_mod_no_weights=context_mod_no_weights,
                 context_mod_post_activation=context_mod_post_activation,
                 context_mod_gain_offset=context_mod_gain_offset, context_mod_gain_softplus=context_mod_gain_softplus,
                 out_fn=out_fn, verbose=verbose)
        
        assert not init_weights, "`init_weights` option is not supported"
        assert not use_context_mod, "`use_context_mod` is not supported"
        assert not use_spectral_norm, "`use_spectral_norm` is not supported"
        # assert (not use_batch_norm) and (not bn_track_stats), "BatchNorm layers are not supported in MLP"

        # Tuple are not mutable.
        hidden_layers = list(hidden_layers)

        self._a_fun = activation_fn
        assert(init_weights is None or \
               (not no_weights or not context_mod_no_weights))
        self._no_weights = no_weights
        self._dropout_rate = dropout_rate
        self._use_batch_norm = use_batch_norm
        self._bn_track_stats = bn_track_stats
        self._out_fn = out_fn

        self._has_bias = use_bias
        self._has_fc_out = True

        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer.
        self._mask_fc_out = True
        self._has_linear_out = True if out_fn is None else False

        self._param_shapes = []
        self._param_shapes_meta = []

        # Initialize lower, middle and upper weights
        if no_weights and context_mod_no_weights:
            self._lower_weights  = None
            self._middle_weights = None
            self._upper_weights  = None
        else:
            self._lower_weights  = nn.ParameterList()
            self._middle_weights = nn.ParameterList()
            self._upper_weights  = nn.ParameterList()
        
        self._hyper_shapes_learned = None \
            if not no_weights and not context_mod_no_weights else []
        self._hyper_shapes_learned_ref = None if self._hyper_shapes_learned \
            is None else []

        if dropout_rate != -1:
            assert(dropout_rate >= 0. and dropout_rate <= 1.)
            self._dropout = IntervalDropout(p=dropout_rate)

        ### Compute shapes of linear layers.
        linear_shapes = MLP.weight_shapes(n_in=n_in, n_out=n_out,
            hidden_layers=hidden_layers, use_bias=use_bias)
        self._param_shapes.extend(linear_shapes)

        for i, s in enumerate(linear_shapes):
            self._param_shapes_meta.append({
                'name': 'weight' if len(s) != 1 else 'bias',
                'index': -1 if no_weights else len(self._middle_weights) + i,
                'layer': -1 # 'layer' is set later.
            })

        num_weights = MainNetInterface.shapes_to_num_weights(self._param_shapes)

        ### Set missing meta information of param_shapes.
        offset = 1 if use_context_mod and context_mod_inputs else 0
        shift = 1
        if use_batch_norm:
            shift += 1
        if use_context_mod:
            shift += 1

        cm_offset = 2 if context_mod_post_activation else 1
        bn_offset = 1 if context_mod_post_activation else 2

        cm_ind = 0
        bn_ind = 0
        layer_ind = 0

        for i, dd in enumerate(self._param_shapes_meta):
            if dd['name'].startswith('cm'):
                if offset == 1 and i in [0, 1]:
                    dd['layer'] = 0
                else:
                    if cm_ind < len(hidden_layers):
                        dd['layer'] = offset + cm_ind * shift + cm_offset
                    else:
                        assert cm_ind == len(hidden_layers) and \
                            not no_last_layer_context_mod
                        # No batchnorm in output layer.
                        dd['layer'] = offset + cm_ind * shift + 1

                    if dd['name'] == 'cm_shift':
                        cm_ind += 1

            elif dd['name'].startswith('bn'):
                dd['layer'] = offset + bn_ind * shift + bn_offset
                if dd['name'] == 'bn_shift':
                        bn_ind += 1

            else:
                dd['layer'] = offset + layer_ind * shift
                if not use_bias or dd['name'] == 'bias':
                    layer_ind += 1

        self._layer_upper_weight_tensors = nn.ParameterList()
        self._layer_middle_weight_tensors = nn.ParameterList()
        self._layer_lower_weight_tensors = nn.ParameterList()

        self._layer_upper_bias_vectors = nn.ParameterList()
        self._layer_middle_bias_vectors = nn.ParameterList()
        self._layer_lower_bias_vectors = nn.ParameterList()

        if no_weights:
            self._hyper_shapes_learned.extend(linear_shapes)

            
            self._is_properly_setup()
            return

        ### Define and initialize linear weights.
        for i, dims in enumerate(linear_shapes):
            self._upper_weights.append(nn.Parameter(torch.Tensor(*dims),
                                              requires_grad=True))
            self._middle_weights.append(nn.Parameter(torch.Tensor(*dims),
                                              requires_grad=True))
            self._lower_weights.append(nn.Parameter(torch.Tensor(*dims),
                                              requires_grad=True))
            
            if len(dims) == 1:
                self._layer_upper_bias_vectors.append(self._upper_weights[-1])
                self._layer_middle_bias_vectors.append(self._middle_weights[-1])
                self._layer_lower_bias_vectors.append(self._lower_weights[-1])
            else:
                self._layer_upper_weight_tensors.append(self._upper_weights[-1])
                self._layer_middle_weight_tensors.append(self._middle_weights[-1])
                self._layer_lower_weight_tensors.append(self._lower_weights[-1])

        
        for i in range(len(self._layer_weight_tensors)):
            if use_bias:
                init_params(self._layer_upper_weight_tensors[i],
                            self._layer_upper_bias_vectors[i])
                
                init_params(self._layer_middle_weight_tensors[i],
                            self._layer_middle_bias_vectors[i])
                
                init_params(self._layer_lower_weight_tensors[i],
                            self._layer_lower_bias_vectors[i])
            else:
                init_params(self._layer_upper_weight_tensors[i])
                init_params(self._layer_middle_weight_tensors[i])
                init_params(self._layer_lower_weight_tensors[i])

        self._is_properly_setup()

    def forward(self, x, upper_weights, middle_weights, lower_weights, condition=None):
        """
        Compute the output y of this network given the input x.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor of shape (batch_size, 3, input_size).
        upper_weights : list of torch.Tensor
            The upper weights of the network.
        middle_weights : list of torch.Tensor
            The middle weights of the network.
        lower_weights : list of torch.Tensor
            The lower weights of the network.
        condition : int, optional
            Needed for application of BatchNorm statistics to test set.

        Returns:
        --------
        tuple
            A tuple containing the output tensor of shape (batch_size, 3, output_size).
        """
        if ((not self._use_context_mod and self._no_weights) or \
                (self._no_weights or self._context_mod_no_weights)) and \
                middle_weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        ############################################
        ### Extract which weights should be used ###
        ############################################
        # I.e., are we using internally maintained weights or externally given
        # ones or are we even mixing between these groups.
        n_cm = self._num_context_mod_shapes()

        int_upper_weights = None
        int_middle_weights = None
        int_lower_weights = None

        if isinstance(upper_weights, dict) and isinstance(middle_weights, dict) and isinstance(lower_weights, dict):
            assert('internal_weights' in upper_weights.keys() or \
                    'mod_weights' in upper_weights.keys())
            
            assert('internal_weights' in middle_weights.keys() or \
                    'mod_weights' in middle_weights.keys())
            
            assert('internal_weights' in lower_weights.keys() or \
                    'mod_weights' in lower_weights.keys())
            
            if 'internal_weights' in upper_weights.keys() and \
                'internal_weights' in middle_weights.keys() and \
                    'internal_weights' in lower_weights.keys():
                
                int_upper_weights = upper_weights['internal_weights']
                int_middle_weights = middle_weights['internal_weights']
                int_lower_weights = lower_weights['internal_weights']
                
            if 'mod_weights' in upper_weights.keys() and \
                'mod_weights' in middle_weights.keys() and \
                'mod_weights' in lower_weights.keys():

                raise Exception("Deprecated!")
        else:
           
            assert(len(upper_weights) == len(self.param_shapes))
            assert(len(middle_weights) == len(self.param_shapes))
            assert(len(lower_weights) == len(self.param_shapes))

            
            int_upper_weights = upper_weights
            int_middle_weights = middle_weights
            int_lower_weights = lower_weights

        if int_upper_weights is None and \
            int_middle_weights is None and \
            int_lower_weights is None:

            if self._no_weights:
                raise Exception('Network was generated without internal ' +
                    'weights. Hence, they must be passed via the ' +
                    '"weights" option.')
            if self._context_mod_no_weights:
                int_upper_weights = self._upper_weights
                int_middle_weights = self._middle_weights
                int_lower_weights = self._lower_weights
            else:
                int_upper_weights = self._upper_weights[n_cm:]
                int_middle_weights = self._middle_weights[n_cm:]
                int_lower_weights = self._lower_weights[n_cm:]

        # Note, context-mod weights might have different shapes, as they
        # may be parametrized on a per-sample basis.
        
        int_shapes = self.param_shapes[n_cm:]

        assert(len(int_upper_weights) == len(int_shapes))
        assert(len(int_middle_weights) == len(int_shapes))
        assert(len(int_lower_weights) == len(int_shapes))

        for i, s in enumerate(int_shapes):
            assert(np.all(np.equal(s, list(int_upper_weights[i].shape))))
            assert(np.all(np.equal(s, list(int_middle_weights[i].shape))))
            assert(np.all(np.equal(s, list(int_lower_weights[i].shape))))

        layer_upper_weights = int_upper_weights
        layer_middle_weights = int_middle_weights
        layer_lower_weights = int_lower_weights

        w_upper_weights = []
        b_upper_weights = []

        w_middle_weights = []
        b_middle_weights = []

        w_lower_weights = []
        b_lower_weights = []

        for i, (p_upper, p_middle, p_lower) in enumerate(zip(
                                                            layer_upper_weights, 
                                                            layer_middle_weights, 
                                                            layer_lower_weights)):
            if self.has_bias and i % 2 == 1:
                b_upper_weights.append(p_upper)
                b_middle_weights.append(p_middle)
                b_lower_weights.append(p_lower)
            else:
                w_upper_weights.append(p_upper)
                w_middle_weights.append(p_middle)
                w_lower_weights.append(p_lower)


        ###########################
        ### Forward Computation ###
        ###########################
        hidden = torch.stack([x, x, x], dim=1)

        for l in range(len(w_middle_weights)):
            w_upper = w_upper_weights[l]
            w_middle = w_middle_weights[l]
            w_lower = w_lower_weights[l]

            if self.has_bias:
                b_upper = b_upper_weights[l]
                b_middle = b_middle_weights[l]
                b_lower = b_lower_weights[l]
            else:
                b_upper = None
                b_middle = None
                b_lower = None

            # Linear layer.
            hidden = IntervalLinear.apply_linear(hidden,
                                                upper_weights=w_upper,
                                                middle_weights=w_middle,
                                                lower_weights=w_lower,
                                                upper_bias=b_upper,
                                                middle_bias=b_middle,
                                                lower_bias=b_lower)

            # Only for hidden layers.
            if l < len(w_middle_weights) - 1:

                # Dropout
                if self._dropout_rate != -1:
                    hidden = self._dropout(hidden)

                # Non-linearity
                if self._a_fun is not None:
                    hidden = self._a_fun(hidden)

        if self._out_fn is not None:
            return self._out_fn(hidden), hidden

        return hidden

    @staticmethod
    def weight_shapes(n_in=1, n_out=1, hidden_layers=[10, 10], use_bias=True):
        """Compute the tensor shapes of all parameters in a fully-connected
        network.

        Parameters:
        -----------
            n_in: int
                Number of inputs.
            n_out: int
                Number of output units.
            hidden_layers: a list of ints
                Each number denoting the size of a hidden layer.
            use_bias: bool
                Whether the FC layers should have biases.

        Returns:
        --------
            A list of list of integers, denoting the shapes of the individual
            parameter tensors.
        """
        shapes = []

        prev_dim = n_in
        layer_out_sizes = hidden_layers + [n_out]
        for i, size in enumerate(layer_out_sizes):
            shapes.append([size, prev_dim])
            if use_bias:
                shapes.append([size])
            prev_dim = size

        return shapes
    
    def _is_properly_setup(self, check_has_bias=True):
        """This method can be used by classes that implement this interface to
        check whether all required properties have been set."""
        assert(self._param_shapes is not None or self._all_shapes is not None)
        if self._param_shapes is None:
            warn('Private member "_param_shapes" should be specified in each ' +
                 'sublcass that implements this interface, since private ' +
                 'member "_all_shapes" is deprecated.', DeprecationWarning)
            self._param_shapes = self._all_shapes

        if self._hyper_shapes is not None or \
                self._hyper_shapes_learned is not None:
            if self._hyper_shapes_learned is None:
                warn('Private member "_hyper_shapes_learned" should be ' +
                     'specified in each sublcass that implements this ' +
                     'interface, since private member "_hyper_shapes" is ' +
                     'deprecated.', DeprecationWarning)
                self._hyper_shapes_learned = self._hyper_shapes
            # FIXME we should actually assert equality if
            # `_hyper_shapes_learned` was not None.
            self._hyper_shapes = self._hyper_shapes_learned

        if self._weights is not None and self._internal_params is None:
            # Note, in the future we might throw a deprecation warning here,
            # once "weights" becomes deprecated.
            self._internal_params = self._weights

        assert self._internal_params is not None or \
               self._hyper_shapes_learned is not None

        if self._hyper_shapes_learned is None and \
                self.hyper_shapes_distilled is None:
            # Note, `internal_params` should only contain trainable weights and
            # not other things like running statistics. Thus, things that are
            # passed to an optimizer.
            assert len(self._internal_params) == len(self._param_shapes)

        if self._param_shapes_meta is None:
            # Note, this attribute was inserted post-hoc.
            # FIXME Warning is annoying, programmers will notice when they use
            # this functionality.
            #warn('Attribute "param_shapes_meta" has not been implemented!')
            pass
        else:
            assert(len(self._param_shapes_meta) == len(self._param_shapes))
            for dd in self._param_shapes_meta:
                assert isinstance(dd, dict)
                assert 'name' in dd.keys() and 'index' in dd.keys() and \
                    'layer' in dd.keys()
                assert dd['name'] is None or \
                       dd['name'] in ['weight', 'bias', 'bn_scale', 'bn_shift',
                                      'cm_scale', 'cm_shift', 'embedding']

                assert isinstance(dd['index'], int)
                if self._internal_params is None:
                    assert dd['index'] == -1
                else:
                    assert dd['index'] == -1 or \
                        0 <= dd['index'] < len(self._internal_params)

                assert isinstance(dd['layer'], int)
                assert dd['layer'] == -1 or dd['layer'] >= 0

        if self._hyper_shapes_learned is not None:
            if self._hyper_shapes_learned_ref is None:
                # Note, this attribute was inserted post-hoc.
                # FIXME Warning is annoying, programmers will notice when they
                # use this functionality.
                #warn('Attribute "hyper_shapes_learned_ref" has not been ' +
                #     'implemented!')
                pass
            else:
                assert isinstance(self._hyper_shapes_learned_ref, list)
                for ii in self._hyper_shapes_learned_ref:
                    assert isinstance(ii, int)
                    assert ii == -1 or 0 <= ii < len(self._param_shapes)

        assert(isinstance(self._has_fc_out, bool))
        assert(isinstance(self._mask_fc_out, bool))
        assert(isinstance(self._has_linear_out, bool))

        assert(self._layer_weight_tensors is not None)
        assert(self._layer_bias_vectors is not None)

        # Note, you should overwrite the `has_bias` attribute if you do not
        # follow this requirement.
        if check_has_bias:
            assert isinstance(self._has_bias, bool)
            if self._has_bias:
                assert len(self._layer_weight_tensors) == \
                       len(self._layer_bias_vectors)

if __name__ == '__main__':
    pass