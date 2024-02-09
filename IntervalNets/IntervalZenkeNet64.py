"""
# Modification of hypnettorch file
# https://hypnettorch.readthedocs.io/en/latest/_modules/hypnettorch/mnets/zenkenet.html#ZenkeNet
# It was necessary to load TinyImageNet which has a different size than CIFAR-100.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hypnettorch.mnets.classifier_interface import Classifier
from hypnettorch.mnets.mnet_interface import MainNetInterface
from hypnettorch.utils.misc import init_params

from IntervalNets.interval_modules import (IntervalConv2d, 
                              IntervalMaxPool2d,
                              IntervalDropout,
                              IntervalLinear)


class ZenkeNet(Classifier):
    """The network consists of four convolutional layers followed by two fully-
    connected layers. See implementation for details.

    ZenkeNet is a network introduced in

        "Continual Learning Through Synaptic Intelligence", Zenke et al., 2017.

    See Appendix for details.

    We use the same network for a fair comparison to the results reported in the
    paper.

    Args:
        in_shape (tuple or list): The shape of an input sample.

            .. note::
                We assume the Tensorflow format, where the last entry
                denotes the number of channels.
        num_classes (int): The number of output neurons. The chosen architecture
            (see ``arch``) will be adopted accordingly.
        verbose (bool): Allow printing of general information about the
            generated network (such as number of weights).
        arch (str): The architecture to be employed. The following options are
            available.

                - ``cifar``: The convolutional network used by Zenke et al.
                  for their proposed split CIFAR-10/100 experiment.
                - ``tiny``: The convolutional network used by Zenke et al.
                  for TinyImageNet experiment.
        no_weights (bool): If set to ``True``, no trainable parameters will be
            constructed, i.e., weights are assumed to be produced ad-hoc
            by a hypernetwork and passed to the :meth:`forward` method.
        init_weights (optional): This option is for convinience reasons.
            The option expects a list of parameter values that are used to
            initialize the network weights. As such, it provides a
            convinient way of initializing a network with a weight draw
            produced by the hypernetwork.
        dropout_rate (float): If ``-1``, no dropout will be applied. Otherwise a
            number between 0 and 1 is expected, denoting the dropout rate.

            Dropout will be applied after the convolutional layers
            (before pooling) and after the first fully-connected layer
            (after the activation function).

            .. note::
                For the FC layer, the dropout rate is doubled.
    """

    _architectures = {
        "cifar": [
            [32, 3, 3, 3],
            [32],
            [32, 32, 3, 3],
            [32],
            [64, 32, 3, 3],
            [64],
            [64, 64, 3, 3],
            [64],
            [512, 2304],
            [512],
            [10, 512],
            [10],
        ],
        "tiny": [
            [64, 3, 3, 3],
            [64],
            [64, 64, 3, 3],
            [64],
            [64, 64, 3, 3],
            [64],
            [64, 64, 3, 3],
            [64],
            [5, 12544],
            [5],
        ],
    }

    def __init__(
        self,
        in_shape=(32, 32, 3),
        num_classes=10,
        verbose=True,
        arch="cifar",
        no_weights=False,
        init_weights=None,
        dropout_rate=0.25,
    ):
        super(ZenkeNet, self).__init__(num_classes, verbose)

        if arch == "cifar":
            assert in_shape[0] == 32 and in_shape[1] == 32
        elif arch == "tiny":
            assert in_shape[0] == 64 and in_shape[1] == 64
        else:
            raise ValueError(
                "Dataset other than CIFAR and TinyImageNet are " "not handled!"
            )
        self._in_shape = in_shape

        self.architecture = arch
        assert self.architecture in ZenkeNet._architectures.keys()
        self._param_shapes = ZenkeNet._architectures[self.architecture]
        self._param_shapes[-2][0] = num_classes
        self._param_shapes[-1][0] = num_classes

        assert init_weights is None or no_weights is False
        self._no_weights = no_weights

        self._use_dropout = dropout_rate != -1

        self._has_bias = True
        self._has_fc_out = True
        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer.
        self._mask_fc_out = True
        # We don't use any output non-linearity.
        self._has_linear_out = True

        self._num_weights = MainNetInterface.shapes_to_num_weights(
            self._param_shapes
        )
        if verbose:
            print(
                "Creating a ZenkeNet with %d weights" % (self._num_weights)
                + (", that uses dropout." if self._use_dropout else ".")
            )

        if self._use_dropout:
            if dropout_rate > 0.5:
                # FIXME not a pretty solution, but we aim to follow the original
                # paper.
                raise ValueError("Dropout rate must be smaller equal 0.5.")
            self._drop_conv = IntervalDropout(p=dropout_rate)
            self._drop_fc1 = IntervalDropout(p=dropout_rate * 2.0)

        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        if no_weights:
            self._weights = None
            self._hyper_shapes_learned = self._param_shapes
            self._hyper_shapes_learned_ref = list(
                range(len(self._param_shapes))
            )
            self._is_properly_setup()
            return

        ### Define and initialize network weights.
        # Each odd entry of this list will contain a weight Tensor and each
        # even entry a bias vector.
        self._weights = nn.ParameterList()

        for i, dims in enumerate(self._param_shapes):
            self._weights.append(
                nn.Parameter(torch.Tensor(*dims), requires_grad=True)
            )

            if i % 2 == 0:
                self._layer_weight_tensors.append(self._weights[i])
            else:
                assert len(dims) == 1
                self._layer_bias_vectors.append(self._weights[i])

        if init_weights is not None:
            assert len(init_weights) == len(self._param_shapes)
            for i in range(len(init_weights)):
                assert np.all(
                    np.equal(
                        list(init_weights[i].shape),
                        list(self._weights[i].shape),
                    )
                )
                self._weights[i].data = init_weights[i]
        else:
            for i in range(len(self._layer_weight_tensors)):
                init_params(
                    self._layer_weight_tensors[i], self._layer_bias_vectors[i]
                )

        self._is_properly_setup()

    def forward(self, x, upper_weights=None, middle_weights=None, lower_weights=None, distilled_params=None, condition=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`.


        Args:
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            x: Input image.

                .. note::
                    We assume the Tensorflow format, where the last entry
                    denotes the number of channels.
            upper_weights: Upper weights generated by a target network
            lower_weights: Lower weights generated by a target network

        Returns:
            y: The output of the network.
        """
        if distilled_params is not None:
            raise ValueError(
                'Parameter "distilled_params" has no '
                + "implementation for this network!"
            )

        if condition is not None:
            raise ValueError(
                'Parameter "condition" has no '
                + "implementation for this network!"
            )

        if self._no_weights and middle_weights is None:
            raise Exception(
                "Network was generated without weights. "
                + 'Hence, "weights" option may not be None.'
            )
        
        if (upper_weights is None and lower_weights is not None) or \
            (upper_weights is not None and lower_weights is None):
            raise Exception("Upper and lower weights must be provided "
                            + "at the same time!")

        if middle_weights is None:
            middle_weights = self._weights
        else:
            shapes = self.param_shapes
            assert len(middle_weights) == len(shapes)
            for i, s in enumerate(shapes):
                assert np.all(np.equal(s, list(middle_weights[i].shape)))

        # Note, implementation aims to follow:
        #     https://git.io/fj8xP
                
        if upper_weights is None and lower_weights is None:

            # first block
            # CIFAR:
            # 32 -> 32 -> 30 -> 15
            # TinyImageNet:
            # 64 -> 64 -> 62 -> 31
            x = x.view(-1, *self._in_shape)
            x = x.permute(0, 3, 1, 2)
            h = F.conv2d(x, middle_weights[0], bias=middle_weights[1], padding=1)  # 'SAME'
            h = F.relu(h)
            h = F.conv2d(h, middle_weights[2], bias=middle_weights[3], padding=0)  # 'VALID'
            # stride and kernel size are equal to 2
            h = F.max_pool2d(F.relu(h), 2)
            if self._use_dropout:
                h = self._drop_conv(h)

            # second block
            # CIFAR
            # 15 -> 15 -> 13 -> 6
            # TinyImageNet
            # 31 -> 31 -> 29 -> 14
            h = F.conv2d(h, middle_weights[4], bias=middle_weights[5], padding=1)  # 'SAME'
            h = F.relu(h)
            h = F.conv2d(h, middle_weights[6], bias=middle_weights[7], padding=0)  # 'VALID'
            h = F.max_pool2d(F.relu(h), 2)
            if self._use_dropout:
                h = self._drop_conv(h)

            # last fully connected layer or layers
            # CIFAR
            # 6 x 6 x 34 = 2304
            # TinyImageNet
            # 14 x 14 x 64 = 12.544
            h = h.reshape(-1, middle_weights[8].size()[1])
            if self.architecture == "cifar":
                h = F.relu(F.linear(h, middle_weights[8], bias=middle_weights[9]))
                if self._use_dropout:
                    h = self._drop_fc1(h)
                h = F.linear(h, middle_weights[10], bias=middle_weights[11])
            elif self.architecture == "tiny":
                h = F.linear(h, middle_weights[8], bias=middle_weights[9])
            return h
        
        else:
            # first block
            # CIFAR:
            # 32 -> 32 -> 30 -> 15
            # TinyImageNet:
            # 64 -> 64 -> 62 -> 31
            x = x.view(-1, *self._in_shape)
            x = x.permute(0, 3, 1, 2)
            x = torch.stack([x, x, x], dim=1)
            h = IntervalConv2d.apply_conv2d(x, 
                                            lower_weights=lower_weights[0],
                                            middle_weights=middle_weights[0],
                                            upper_weights=upper_weights[0],
                                            lower_bias=lower_weights[1],
                                            middle_bias=middle_weights[1],
                                            upper_bias=upper_weights[1],
                                            padding=1)  # 'SAME'
            h = F.relu(h)
            h = IntervalConv2d.apply_conv2d(h, 
                                            lower_weights=lower_weights[2],
                                            middle_weights=middle_weights[2],
                                            upper_weights=upper_weights[2],
                                            lower_bias=lower_weights[3],
                                            middle_bias=middle_weights[3],
                                            upper_bias=upper_weights[3],
                                            padding=0)  # 'VALID'
            
            # stride and kernel size are equal to 2
            
            h = IntervalMaxPool2d.apply_max_pool2d(F.relu(h), 2)
            
            if self._use_dropout:
                h = self._drop_conv(h)

            # second block
            # CIFAR
            # 15 -> 15 -> 13 -> 6
            # TinyImageNet
            # 31 -> 31 -> 29 -> 14
            h = IntervalConv2d.apply_conv2d(h, 
                                            lower_weights=lower_weights[4],
                                            middle_weights=middle_weights[4],
                                            upper_weights=upper_weights[4],
                                            lower_bias=lower_weights[5],
                                            middle_bias=middle_weights[5],
                                            upper_bias=upper_weights[5],
                                            padding=1)  # 'SAME'
            h = F.relu(h)
            
            h = IntervalConv2d.apply_conv2d(h, 
                                            lower_weights=lower_weights[6],
                                            middle_weights=middle_weights[6],
                                            upper_weights=upper_weights[6],
                                            lower_bias=lower_weights[7],
                                            middle_bias=middle_weights[7],
                                            upper_bias=upper_weights[7],
                                            padding=0)  # 'SAME'  # 'VALID'
            h = IntervalMaxPool2d.apply_max_pool2d(F.relu(h), 2)
            if self._use_dropout:
                h = self._drop_conv(h)

            # last fully connected layer or layers
            # CIFAR
            # 6 x 6 x 34 = 2304
            # TinyImageNet
            # 14 x 14 x 64 = 12.544
            h = h.rename(None)
            h = h.reshape([-1, 3, middle_weights[8].shape[1]])

            if self.architecture == "cifar":

                h = IntervalLinear.apply_linear(h, 
                                                lower_weights=lower_weights[8],
                                                middle_weights=middle_weights[8],
                                                upper_weights=upper_weights[8],
                                                lower_bias=lower_weights[9],
                                                middle_bias=middle_weights[9],
                                                upper_bias=upper_weights[9])
                h = F.relu(h)
                if self._use_dropout:
                    h = self._drop_fc1(h)

                h = IntervalLinear.apply_linear(h, 
                                                lower_weights=lower_weights[10],
                                                middle_weights=middle_weights[10],
                                                upper_weights=upper_weights[10],
                                                lower_bias=lower_weights[11],
                                                middle_bias=middle_weights[11],
                                                upper_bias=upper_weights[11])
            elif self.architecture == "tiny":
                h = IntervalLinear.apply_linear(h, 
                                                lower_weights=lower_weights[8],
                                                middle_weights=middle_weights[8],
                                                upper_weights=upper_weights[8],
                                                lower_bias=lower_weights[9],
                                                middle_bias=middle_weights[9],
                                                upper_bias=upper_weights[9])
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