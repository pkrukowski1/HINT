# Copyright 2020 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :mnets/lenet.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :02/28/2020
# @version        :1.0
# @python_version :3.6.10
"""
LeNet 300-100
-----

This module contains LeNet 300-100 network to classify MNIST images. 
The network 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hypnettorch.mnets.classifier_interface import Classifier
from hypnettorch.utils.torch_utils import init_params

class LeNet(Classifier):
    """The network consists of the following layers:
        I. Feature extractor:
            (1) Convolutional layer:
                - number of input channels: 1
                - number of output channels: 64
                - kernel size: (3,3)
                - stride: 1
                - padding: 1
            (2) ReLU activation function
            (3) Concolutional layer:
                - number of input channels: 64
                - number of output channels: 64
                - kernel size: (3,3)
                - stride: 1
                - padding: 1
            (4) ReLU activation function
            (5) Max pooling layer:
                - kernel size: 2
        II. Classifier:
            (0) Flatten
            (1) Linear layer:
                - number of input features: 64*14**2 (assuming input_size = [28,28,1])
                - number of output features: 300
            (2) Relu activaton function
            (3) Linear layer:
                - number of input features: 300
                - number of output features: 100
            (4) ReLU activation function
            (5) Linear layer (final)
                - number of input features: 100
                - number of output features: 10

    LeNet was originally introduced in

        "Gradient-based learning applied to document recognition", LeCun et
        al., 1998.

    Parameters:
    -----------
        in_shape: Tuple|List
        --------
            The shape of an input sample.
            .. note::
                We assume the Tensorflow format, where the last entry
                denotes the number of channels.
        num_classes: int
        -----------
            The number of output neurons.
        verbose: bool
        -------
            Allow printing of general information about the
            generated network (such as number of weights).
        init_weights: Optional
        ------------
            This option is for convinience reasons.
            The option expects a list of parameter values that are used to
            initialize the network weights. As such, it provides a
            convinient way of initializing a network with a weight draw
            produced by the hypernetwork.
    """
    _ARCHITECTURE = [[64,1,3,3],[64],   # First conv layer
                     [64,64,3,3],[64],  # Second conv layer
                     [300,3136],[300], # First linear layer
                     [100,300],[100],   # Second linear layer
                     [10,100],[10]]     # Classifier head layer

    def __init__(self, in_shape=(28, 28, 1), num_classes=10, verbose=True,
                 no_weights=True, init_weights=None):
        
        assert no_weights == True, "No additional trainable parameters should be added."

        super(LeNet, self).__init__(num_classes, verbose)

        self._in_shape = in_shape
        self._arch = LeNet._ARCHITECTURE
        if num_classes != 10:
            self._arch[-2][0] = num_classes
            self._arch[-1][0] = num_classes

        # Sanity check, given current implementation.
        if not in_shape[0] == in_shape[1] == 28:
                raise ValueError('MNIST LeNet architectures expect input ' +
                                 'images of size 28x28.')

        ### Setup class attributes ###
        self._no_weights = True

        self._has_bias = True
        self._has_fc_out = True
        # We need to make sure that the last 2 entries of `weights` correspond
        # to the weight matrix and bias vector of the last layer!
        self._mask_fc_out = True
        self._has_linear_out = True

        self._param_shapes = []
        self._param_shapes_meta = []
        self._internal_params = None
        self._hyper_shapes_learned = []
        self._hyper_shapes_learned_ref = None if self._hyper_shapes_learned \
            is None else []
        self._layer_weight_tensors = nn.ParameterList()
        self._layer_bias_vectors = nn.ParameterList()

        ### Define and add conv- and fc-layer weights.
        for i, s in enumerate(self._arch):
            self._hyper_shapes_learned.append(s)
            self._hyper_shapes_learned_ref.append(len(self.param_shapes))

            self._param_shapes.append(s)
            self._param_shapes_meta.append({
                'name': 'weight' if len(s) != 1 else 'bias',
                'index': -1 if no_weights else len(self._internal_params)-1,
                'layer': 2 * (i // 2) + 1
            })

        ### Initialize weights.
        if init_weights is not None:
            assert len(init_weights) == len(self.weights)
            for i in range(len(init_weights)):
                assert np.all(np.equal(list(init_weights[i].shape),
                                       self.weights[i].shape))
                self.weights[i].data = init_weights[i]
        else:
            for i in range(len(self._layer_weight_tensors)):
                init_params(self._layer_weight_tensors[i],
                            self._layer_bias_vectors[i])

        ### Print user info.
        if verbose:
            print('Creating a LeNet with %d weights' % self.num_params
                  + '.'
                 )

        self._is_properly_setup()

    def forward(self, x, weights, condition=None):
        """Compute the output :math:`y` of this network given the input
        :math:`x`. Please note that `condition` is used just to match
        a target network implementation for other datasets.

        Parameters:
        -----------
            (....): See docstring of method
                :meth:`mnets.mnet_interface.MainNetInterface.forward`. We
                provide some more specific information below.
            weights: List | Dict
            --------
                See argument ``weights`` of method :meth:`mnets.mlp.MLP.forward`.

        Returns:
        --------
            (torch.Tensor): The output of the network.
        """
        
        assert weights is not None, "There is required to provide weights of the target network."

        ############################################
        ### Extract which weights should be used ###
        ############################################
        # FIXME Code copied from MLP its `forward` method.
        # I.e., are we using internally maintained weights or externally given
        # ones or are we even mixing between these groups.
        n_cm = self._num_context_mod_shapes()

        int_weights = None

        if isinstance(weights, dict):
            assert('internal_weights' in weights.keys() or \
                    'mod_weights' in weights.keys())
            if 'internal_weights' in weights.keys():
                int_weights = weights['internal_weights']
        else:
            int_weights = weights

        if int_weights is None:
            if self._no_weights:
                raise Exception('Network was generated without internal ' +
                    'weights. Hence, they must be passed via the ' +
                    '"weights" option.')
            int_weights = self.weights[n_cm:]

        # Note, context-mod weights might have different shapes, as they
        # may be parametrized on a per-sample basis.
        int_shapes = self.param_shapes[n_cm:]
        assert len(int_weights) == len(int_shapes)
        for i, s in enumerate(int_shapes):
            assert np.all(np.equal(s, list(int_weights[i].shape)))


        ###########################
        ### Forward Computation ###
        ###########################

        ### Feature extractor
        x = x.view(-1, *self._in_shape)
        x = x.permute(0, 3, 1, 2)
        h = x

        # First convolutional layer + activation function
        h = F.conv2d(
            input=h,
            weight=int_weights[0],
            bias=int_weights[1],
            stride=1,
            padding=1
        )
        h = torch.relu(h)
        h = F.max_pool2d(h,2)

        # Second convolutional layer + activation function + max pool layer
        h = F.conv2d(input=h,
                     weight=int_weights[2],
                     bias=int_weights[3],
                     stride=1,
                     padding=1)
        h = torch.relu(h)
        h = F.max_pool2d(h, 2)

        ### Flatten the output from the feature extractor
        h = h.reshape(-1, int_weights[4].size()[1])

        ### Classifier

        # First linear layer + activaton function
        h = F.linear(h, int_weights[4], bias=int_weights[5])
        h = torch.relu(h)

        # Second linear layer + activation function
        h = F.linear(h, int_weights[6], bias=int_weights[7])
        h = torch.relu(h)

        # Classifier head
        h = F.linear(h, int_weights[8], bias=int_weights[9])     

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



if __name__ == '__main__':
    pass