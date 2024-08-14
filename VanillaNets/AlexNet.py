import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hypnettorch.mnets.classifier_interface import Classifier
from hypnettorch.mnets.mnet_interface import MainNetInterface


class AlexNet(Classifier):
    """Implementation of AlexNet to make a fair comparison between
    InterContiNet and HyperInterval results.

    Only CIFAR-10/100 datasets are supported right now.

    Parameters:
    -----------
        in_shape: tuple or list
            The shape of an input sample.
            .. note::
                We assume the Tensorflow format, where the last entry
                denotes the number of channels.
        num_classes: int
            The number of output neurons.
        arch: str
            A neural network architecture. Only CIFAR-10/100 is supported
            right now. The only possible default value is 'cifar'.
        verbose: bool
            Allow printing of general information about the
            generated network (such as number of weights).
        no_weights: bool
            If set to ``True``, no trainable parameters will be
            constructed, i.e., weights are assumed to be produced ad-hoc
            by a hypernetwork and passed to the :meth:`forward` method.
        init_weights: optional
            This option is for convinience reasons.
            The option expects a list of parameter values that are used to
            initialize the network weights. As such, it provides a
            convinient way of initializing a network with a weight draw
            produced by the hypernetwork.
        bn_track_stats: bool
            If is set to False, this layer then does not keep running estimates
            and batch statistics are instead used during evaluation time as well. For more information
            please see docs https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        distill_bn_stats: bool
        If ``True``, then the shapes of the batchnorm statistics will be added to the attribute
            :attr:`mnets.mnet_interface.MainNetInterface.hyper_shapes_distilled` and the current statistics 
            will be returned by the method :meth:`distillation_targets`. Currently it is not used.

    Returns:
    --------
        torch.Tensor
    """

    def __init__(
        self,
        in_shape=(32, 32, 3),
        num_classes=10,
        verbose=True,
        arch="cifar",
        no_weights=True,
        use_batch_norm=True,
        bn_track_stats=True,
        distill_bn_stats=False,
        init_weights=None,
    ):
        super(AlexNet, self).__init__(num_classes, verbose)

        _architectures = {
        "cifar": [
            [64, 3, 3, 3],
            [64],
            [192, 64, 3, 3],
            [192],
            [384, 192, 3, 3],
            [384],
            [256, 384, 3, 3],
            [256],
            [256, 256, 3, 3],
            [256],
            [4096, 256*2*2],
            [4096],
            [4096, 4096],
            [4096],
            [num_classes, 4096],
            [num_classes]
            ]
        }

        if arch == "cifar":
            assert in_shape[0] == 32 and in_shape[1] == 32
        else:
            raise ValueError(
                "Dataset other than CIFAR are " "not handled!"
            )
        self._in_shape = in_shape

        self._hyper_shapes_learned = (
            None if not no_weights and not self._context_mod_no_weights else []
        )
        self._hyper_shapes_learned_ref = (
            None if self._hyper_shapes_learned is None else []
        )

        self.architecture = arch
        assert self.architecture in _architectures.keys()
        self._param_shapes = _architectures[self.architecture]
        self._param_shapes[-2][0] = num_classes
        self._param_shapes[-1][0] = num_classes

        assert init_weights is None or no_weights is False
        self._no_weights = no_weights

        self._has_bias = True
        self._has_fc_out = True
        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer.
        self._mask_fc_out = True
        # We don't use any output non-linearity.
        self._has_linear_out = True
        self._use_batch_norm = use_batch_norm
        self._bn_track_stats = bn_track_stats

        # Add BatchNorm layers at the end
        if use_batch_norm:
            
            start_idx = len(_architectures[self.architecture])

            bn_sizes = [
                64, 192, 384, 256, 256, 4096, 4096
            ]

            bn_layers = list(range(start_idx, start_idx + len(bn_sizes)))
            self._bn_params_start_idx = start_idx

            self._add_batchnorm_layers(
                bn_sizes,
                no_weights,
                bn_layers=bn_layers,
                distill_bn_stats=distill_bn_stats,
                bn_track_stats=bn_track_stats,
            )

        self._num_weights = MainNetInterface.shapes_to_num_weights(
            self._param_shapes
        )
        if verbose:
            print(
                "Creating an AlexNet with %d weights" % (self._num_weights)
            )

        self._weights = None
        self._hyper_shapes_learned = self._param_shapes
        self._hyper_shapes_learned_ref = list(
            range(len(self._param_shapes))
        )

        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        self._is_properly_setup()

    def forward(self, x, weights=None, distilled_params=None, condition=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.


        Parameters:
        -----------
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            x: torch.Tensor
                Input image.

                .. note::
                    We assume the Tensorflow format, where the last entry
                    denotes the number of channels.

        Returns:
        ---------
            y: torch.Tensor
                The output of the network.
        """

        if distilled_params is not None:
            raise ValueError(
                'Parameter "distilled_params" has no '
                + "implementation for this network!"
            )
        
        if self._no_weights and weights is None:
            raise Exception(
                "Network was generated without weights. "
                + 'Hence, "weights" option may not be None.'
            )
        
        # Parse conditions
        bn_cond = None

        if condition is not None:
            if isinstance(condition, dict):
                assert (
                    "bn_stats_id" in condition.keys()
                    or "cmod_ckpt_id" in condition.keys()
                )
                if "bn_stats_id" in condition.keys():
                    bn_cond = condition["bn_stats_id"]
                if "cmod_ckpt_id" in condition.keys():
                    raise ValueError("Context modulation layers are not used!")
            else:
                bn_cond = condition


        if weights is None:
            weights = self._weights
        else:
            shapes = self.param_shapes
            assert len(weights) == len(shapes)
            for i, s in enumerate(shapes):
                assert np.all(np.equal(s, list(weights[i].shape)))


        ######################################
        ### Select batchnorm running stats ###
        ######################################
        if self._use_batch_norm:
            nn = len(self._batchnorm_layers)
            running_means = [None] * nn
            running_vars = [None] * nn

        if self._use_batch_norm and self._bn_track_stats and bn_cond is None:
            for i, bn_layer in enumerate(self._batchnorm_layers):
                running_means[i], running_vars[i] = bn_layer.get_stats()

        bn_params_start_idx = self._bn_params_start_idx
        
        ########################
        ### Forward computations
        ########################

        x = x.view(-1, *self._in_shape)
        x = x.permute(0, 3, 1, 2)

        ### Convolutional layers

        # First convolutional block + pooling
        h = F.conv2d(x, weights[0], bias=weights[1], stride=2, padding=1)
        h = F.max_pool2d(h, kernel_size=2)
        h = F.relu(h)

        # Batch normalization
        if self._use_batch_norm:
            h = self._batchnorm_layers[0].forward(
                        h,
                        running_mean=running_means[0],
                        running_var=running_vars[0],
                        weight=weights[bn_params_start_idx],
                        bias=weights[bn_params_start_idx+1],
                        stats_id=bn_cond,
                    )

        # Second convolutional block
        h = F.conv2d(h, weights[2], bias=weights[3], padding=1)
        h = F.max_pool2d(h, kernel_size=2)
        h = F.relu(h)

        # Batch normalization
        if self._use_batch_norm:
            h = self._batchnorm_layers[1].forward(
                        h,
                        running_mean=running_means[1],
                        running_var=running_vars[1],
                        weight=weights[bn_params_start_idx+2],
                        bias=weights[bn_params_start_idx+3],
                        stats_id=bn_cond,
                    )

        # Third convolutional block
        h = F.conv2d(h, weights[4], bias=weights[5], padding=1)
        h = F.relu(h)

        # Batch normalization
        if self._use_batch_norm:
            h = self._batchnorm_layers[2].forward(
                        h,
                        running_mean=running_means[2],
                        running_var=running_vars[2],
                        weight=weights[bn_params_start_idx+4],
                        bias=weights[bn_params_start_idx+5],
                        stats_id=bn_cond,
                    )

        # Fourth convolutional block
        h = F.conv2d(h, weights[6], bias=weights[7], padding=1)
        h = F.relu(h)

        # Batch normalization
        if self._use_batch_norm:
            h = self._batchnorm_layers[3].forward(
                        h,
                        running_mean=running_means[3],
                        running_var=running_vars[3],
                        weight=weights[bn_params_start_idx+6],
                        bias=weights[bn_params_start_idx+7],
                        stats_id=bn_cond,
                    )

        # Fifth convolutional block
        h = F.conv2d(h, weights[8], bias=weights[9], padding=1)
        h = F.max_pool2d(h, kernel_size=2)
        h = F.relu(h)

        # Batch normalization
        if self._use_batch_norm:
            h = self._batchnorm_layers[4].forward(
                        h,
                        running_mean=running_means[4],
                        running_var=running_vars[4],
                        weight=weights[bn_params_start_idx+8],
                        bias=weights[bn_params_start_idx+9],
                        stats_id=bn_cond,
                    )

        ### Fully-connected layers
        h = h.reshape(-1, weights[10].size()[1])
        
        # First fully-connected layer
        h = F.linear(h, weights[10], bias=weights[11])
        h = F.relu(h)

        # Batch normalization
        if self._use_batch_norm:
            h = self._batchnorm_layers[5].forward(
                        h,
                        running_mean=running_means[5],
                        running_var=running_vars[5],
                        weight=weights[bn_params_start_idx+10],
                        bias=weights[bn_params_start_idx+11],
                        stats_id=bn_cond,
                    )

        # Second fully-connected layer
        h = F.linear(h, weights[12], bias=weights[13])
        h = F.relu(h)

        # Batch normalization
        if self._use_batch_norm:
            h = self._batchnorm_layers[6].forward(
                        h,
                        running_mean=running_means[6],
                        running_var=running_vars[6],
                        weight=weights[bn_params_start_idx+12],
                        bias=weights[bn_params_start_idx+13],
                        stats_id=bn_cond,
                    )

        # Third fully-connected layer (output layer)
        h = F.linear(h, weights[14], bias=weights[15])

        return h

    def distillation_targets(self):
        """Targets to be distilled after training.

        See docstring of abstract super method
        :meth:`mnets.mnet_interface.MainNetInterface.distillation_targets`.

        This network does not have any distillation targets.

        Returns:
            ``None``
        """
        return None