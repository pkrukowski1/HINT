# It is code based on the file https://github.com/gmum/InterContiNet/blob/main/intervalnet/models/interval.py
# licensed under the MIT License

from typing import cast
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def parse_logits(x):
    """
    Parse the output of a target network to get lower, middle and upper predictions

    Parameters:
    ----------
        x: torch.Tensor
          The tensor of shape (batch_size, 3, tensor_dimension) to be parsed
    
    Returns:
    --------
        A tuple of lower, middle and upper predictions
    """

    return map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore

class IntervalModuleWithWeights(nn.Module, ABC):
    def __init__(self):
        super().__init__()

class IntervalLinear(IntervalModuleWithWeights):
    """
    Interval linear layer with weights.

    Parameters:
    -----------
        in_features: int
            Number of input features.
        out_features: int
            Number of output features.

    Attributes:
        in_features: int
            Number of input features.
        out_features: int
            Number of output features.

    Methods:
        apply_linear(
            x: torch.Tensor,
            upper_weights: torch.Tensor,
            middle_weights: torch.Tensor,
            lower_weights: torch.Tensor,
            upper_bias: torch.Tensor,
            middle_bias: torch.Tensor,
            lower_bias: torch.Tensor
        ) -> torch.Tensor:
            Computes the output bounds based on weights and biases.

        Returns:
            torch.Tensor:
                Output tensor with bounds (batch_size, bounds, features).

        Raises:
            AssertionError:
                If input bounds violate constraints.
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        nn.Module.__init__()

        self.in_features  = in_features
        self.out_features = out_features
       
    @staticmethod
    def apply_linear( 
                x: Tensor, 
                upper_weights: Tensor,
                middle_weights: Tensor,
                lower_weights: Tensor,
                upper_bias: Tensor,
                middle_bias: Tensor,
                lower_bias: Tensor
                ) -> Tensor:  # type: ignore
        """
        Computes the output bounds based on weights and biases.

        Parameters:
        -----------
            x: torch.Tensor
                Input tensor with shape (batch_size, bounds, features).
            upper_weights: torch.Tensor
                Upper weights with shape (out_features, in_features).
            middle_weights: torch.Tensor
                Middle weights with shape (out_features, in_features).
            lower_weights: torch.Tensor
                Lower weights with shape (out_features, in_features).
            upper_bias: torch.Tensor
                Upper bias with shape (out_features).
            middle_bias: torch.Tensor
                Middle bias with shape (out_features).
            lower_bias: torch.Tensor
                Lower bias with shape (out_features).

        Returns:
            torch.Tensor:
                Output tensor with bounds (batch_size, bounds, features).

        Raises:
            AssertionError:
                If input bounds violate constraints.
        """

        assert (lower_weights <= middle_weights).all(), "Lower bound must be less than or equal to middle bound."
        assert (middle_weights <= upper_weights).all(), "Middle bound must be less than or equal to upper bound."
        assert (lower_bias <= middle_bias).all(), "Lower bias must be less than or equal to middle bias."
        assert (middle_bias <= upper_bias).all(), "Middle bias must be less than or equal to upper bias."

        x = x.refine_names("N", "bounds", "features")  # type: ignore
        assert (x.rename(None) >= 0.0).all(), "All input features must be non-negative."  # type: ignore

        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        assert (x_lower <= x_middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (x_middle <= x_upper).all(), "Middle bound must be less than or equal to upper bound."


        w_lower_pos = lower_weights.clamp(min=0)
        w_lower_neg = lower_weights.clamp(max=0)
        w_upper_pos = upper_weights.clamp(min=0)
        w_upper_neg = upper_weights.clamp(max=0)

        # Further splits only needed for numeric stability with asserts
        w_middle_pos = middle_weights.clamp(min=0)
        w_middle_neg = middle_weights.clamp(max=0)

        lower = x_lower @ w_lower_pos.t() + x_upper @ w_lower_neg.t()
        upper = x_upper @ w_upper_pos.t() + x_lower @ w_upper_neg.t()
        middle = x_middle @ w_middle_pos.t() + x_middle @ w_middle_neg.t()

        b_middle = middle_bias
        b_lower = lower_bias
        b_upper = upper_bias
        lower = lower + b_lower
        upper = upper + b_upper
        middle = middle + b_middle

        assert (lower <= middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (middle <= upper).all(), "Middle bound must be less than or equal to upper bound."

        return torch.stack([lower, middle, upper], dim=1).refine_names("N", "bounds", "features")  # type: ignore
        

class IntervalDropout(nn.Module):
    def __init__(self, p=0.5):
        """
        Initializes an IntervalDropout layer.

        Parameters:
        -----------
            p (float): The probability of dropping an interval. Default is 0.5.
        """
         
        super().__init__()
        self.p = p
        self.scale = 1. / (1 - self.p)

    def forward(self, x):
        """
        Applies IntervalDropout to the input tensor.

        Parameters:
        -----------
            x (torch.Tensor): Input tensor with named dimensions "N", "bounds", ...

        Returns:
            torch.Tensor: Output tensor after applying IntervalDropout.
        """
        if self.training:
            x = x.refine_names("N", "bounds", ...)
            x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)),
                                             x.unbind("bounds"))  # type: ignore
            mask = torch.bernoulli(self.p * torch.ones_like(x_middle)).long()
            x_lower = x_lower.where(mask != 1, torch.zeros_like(x_lower)) * self.scale
            x_middle = x_middle.where(mask != 1, torch.zeros_like(x_middle)) * self.scale
            x_upper = x_upper.where(mask != 1, torch.zeros_like(x_upper)) * self.scale

            return torch.stack([x_lower, x_middle, x_upper], dim=1)
        else:
            return x


class IntervalMaxPool2d(nn.MaxPool2d):
    """
    Initializes an IntervalMaxPool2d layer.

    Parameters:
    -----------
    kernel_size : int or tuple
        Size of the max pooling window.
    stride : int or tuple, optional
        Stride of the max pooling window. Default is None.
    padding : int or tuple, optional
        Padding added to each dimension of the input. Default is 0.
    dilation : int or tuple, optional
        Spacing between kernel elements. Default is 1.
    return_indices : bool, optional
        If True, return the max indices along with the output. Default is False.
    ceil_mode : bool, optional
        If True, use ceil instead of floor to compute output shape. Default is False.
    """
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x):
        """
        Applies IntervalMaxPool2d to the input tensor.

        Parameters:
        -----------
            x (torch.Tensor): Input tensor with named dimensions "N", "bounds", ...

        Returns:
            torch.Tensor: Output tensor after applying IntervalMaxPool2d.
        """
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        x_lower = super().forward(x_lower)
        x_middle = super().forward(x_middle)
        x_upper = super().forward(x_upper)

        return torch.stack([x_lower, x_middle, x_upper], dim=1).refine_names("N", "bounds", "C", "H", "W")  # type: ignore

    @staticmethod
    def apply_max_pool2d(x, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        """
        Applies max pooling to the input tensor.

        Parameters:
        -----------
            x (torch.Tensor): Input tensor with named dimensions "N", "bounds", ...

        Returns:
            torch.Tensor: Output tensor after applying max pooling.
        """
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore

        x_lower = F.max_pool2d(x_lower, kernel_size, stride=stride, padding=padding,
                               dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
        x_middle = F.max_pool2d(x_middle, kernel_size, stride=stride, padding=padding,
                                dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
        x_upper = F.max_pool2d(x_upper, kernel_size, stride=stride, padding=padding,
                               dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)

        return torch.stack([x_lower, x_middle, x_upper], dim=1).refine_names("N", "bounds", "C", "H", "W")  # type: ignore



class IntervalAvgPool2d(nn.AvgPool2d):
    """
    Initializes an IntervalAvgPool2d layer.

    Parameters:
    -----------
    kernel_size : int or tuple
        Size of the average pooling window.
    stride : int or tuple, optional
        Stride of the average pooling window. Default is None.
    padding : int or tuple, optional
        Padding added to each dimension of the input. Default is 0.
    ceil_mode : bool, optional
        If True, use ceil instead of floor to compute output shape. Default is False.
    count_include_pad : bool, optional
        If True, include zero-padding in the averaging. Default is True.
    divisor_override : int or None, optional
        If not None, use this value as divisor for averaging. Default is None.

    """
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    def forward(self, x):
        """
        Applies IntervalAvgPool2d to the input tensor.

        Parameters:
        -----------
            x (torch.Tensor): Input tensor with named dimensions "N", "bounds", ...

        Returns:
            torch.Tensor: Output tensor after applying IntervalAvgPool2d.
        """
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        x_lower = super().forward(x_lower)
        x_middle = super().forward(x_middle)
        x_upper = super().forward(x_upper)

        return torch.stack([x_lower, x_middle, x_upper], dim=1).refine_names("N", "bounds", "C", "H", "W")  # type: ignore

    @staticmethod
    def apply_avg_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                         divisor_override=None):
        """
        Applies average pooling to the input tensor.

        Parameters:
        -----------
            x (torch.Tensor): Input tensor with named dimensions "N", "bounds", ...

        Returns:
            torch.Tensor: Output tensor after applying average pooling.
        """
        x = x.refine_names("N", "bounds", ...)
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore

        x_lower = F.avg_pool2d(x_lower, kernel_size, stride=stride, padding=padding,
                               ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)
        x_middle = F.avg_pool2d(x_middle, kernel_size, stride=stride, padding=padding,
                                ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)
        x_upper = F.avg_pool2d(x_upper, kernel_size, stride=stride, padding=padding,
                               ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)

        return torch.stack([x_lower, x_middle, x_upper], dim=1).refine_names("N", "bounds", "C", "H", "W")  # type: ignore

    


class IntervalConv2d(nn.Conv2d, IntervalModuleWithWeights):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            lower_weights: Tensor,
            middle_weights: Tensor,
            upper_weights: Tensor,
            lower_bias: Tensor,
            middle_bias: Tensor,
            upper_bias: Tensor,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True
    ) -> None:
        
        """
        Initializes an IntervalConv2d layer.

        Parameters:
        -----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Size of the convolutional kernel.
        lower_weights : torch.Tensor
            Lower bound weights for the convolutional layer.
        middle_weights : torch.Tensor
            Middle bound weights for the convolutional layer.
        upper_weights : torch.Tensor
            Upper bound weights for the convolutional layer.
        lower_bias : torch.Tensor
            Lower bound bias for the convolutional layer.
        middle_bias : torch.Tensor
            Middle bound bias for the convolutional layer.
        upper_bias : torch.Tensor
            Upper bound bias for the convolutional layer.
        stride : int, optional
            Stride of the convolution. Default is 1.
        padding : int, optional
            Padding added to each dimension of the input. Default is 0.
        dilation : int, optional
            Spacing between kernel elements. Default is 1.
        groups : int, optional
            Number of blocked connections from input channels to output channels. Default is 1.
        bias : bool, optional
            If True, include bias terms. Default is True.
        """

        IntervalModuleWithWeights.__init__(self)
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.lower_weights  = lower_weights
        self.middle_weights = middle_weights
        self.upper_weights  = upper_weights

        self.lower_bias  = lower_bias
        self.middle_bias = middle_bias
        self.upper_bias  = upper_bias
    
    @staticmethod
    def apply_conv2d(x: Tensor,
                    lower_weights: Tensor,
                    middle_weights: Tensor,
                    upper_weights: Tensor,
                    lower_bias: Tensor,
                    middle_bias: Tensor,
                    upper_bias: Tensor,
                    stride: int = 1,
                    padding: int = 0,
                    dilation: int = 1,
                    groups: int = 1,
                    bias: bool = True) -> Tensor:  # type: ignore
        
        """
        Applies interval convolution to the input tensor.

        Parameters:
        -----------
            x (torch.Tensor): Input tensor with named dimensions "N", "bounds", "C", "H", "W".
            lower_weights (torch.Tensor): Lower bound weights for the convolutional layer.
            middle_weights (torch.Tensor): Middle bound weights for the convolutional layer.
            upper_weights (torch.Tensor): Upper bound weights for the convolutional layer.
            lower_bias (torch.Tensor): Lower bound bias for the convolutional layer.
            middle_bias (torch.Tensor): Middle bound bias for the convolutional layer.
            upper_bias (torch.Tensor): Upper bound bias for the convolutional layer.
            stride (int, optional): Stride of the convolution. Default is 1.
            padding (int, optional): Padding added to each dimension of the input. Default is 0.
            dilation (int, optional): Spacing between kernel elements. Default is 1.
            groups (int, optional): Number of blocked connections from input channels to output channels. Default is 1.
            bias (bool, optional): If True, include bias terms. Default is True.

        Returns:
            torch.Tensor: Output tensor after applying interval convolution.
        """

        x = x.refine_names("N", "bounds", "C", "H", "W")
        assert (x.rename(None) >= 0.0).all(), "All input features must be non-negative."  # type: ignore
        x_lower, x_middle, x_upper = map(lambda x_: cast(Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore
        assert (x_lower <= x_middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (x_middle <= x_upper).all(), "Middle bound must be less than or equal to upper bound."

      
        w_middle: Tensor = middle_weights
        w_lower  = lower_weights
        w_upper  = upper_weights
        b_middle = middle_bias
        b_lower  = lower_bias
        b_upper  = upper_bias

        w_lower_pos = w_lower.clamp(min=0)
        w_lower_neg = w_lower.clamp(max=0)
        w_upper_pos = w_upper.clamp(min=0)
        w_upper_neg = w_upper.clamp(max=0)

        # Further splits only needed for numeric stability with asserts
        w_middle_neg = w_middle.clamp(max=0)
        w_middle_pos = w_middle.clamp(min=0)

        l_lp = F.conv2d(x_lower, w_lower_pos, None, stride, padding, dilation, groups)
        u_ln = F.conv2d(x_upper, w_lower_neg, None, stride, padding, dilation, groups)
        u_up = F.conv2d(x_upper, w_upper_pos, None, stride, padding, dilation, groups)
        l_un = F.conv2d(x_lower, w_upper_neg, None, stride, padding, dilation, groups)
        m_mp = F.conv2d(x_middle, w_middle_pos, None, stride, padding, dilation, groups)
        m_mn = F.conv2d(x_middle, w_middle_neg, None, stride, padding, dilation, groups)

        lower = l_lp + u_ln
        upper = u_up + l_un
        middle = m_mp + m_mn

        if bias is not None and b_lower is not None and \
            b_middle is not None and \
            b_upper is not None:
            
            lower = lower + b_lower.view(1, b_lower.size(0), 1, 1)
            upper = upper + b_upper.view(1, b_upper.size(0), 1, 1)
            middle = middle + b_middle.view(1, b_middle.size(0), 1, 1)

        # Safety net for rare numerical errors.
        if not (lower <= middle).all():
            diff = torch.where(lower > middle, lower - middle, torch.zeros_like(middle)).abs().sum()
            print(f"Lower bound must be less than or equal to middle bound. Diff: {diff}")
            lower = torch.where(lower > middle, middle, lower)
        if not (middle <= upper).all():
            diff = torch.where(middle > upper, middle - upper, torch.zeros_like(middle)).abs().sum()
            print(f"Middle bound must be less than or equal to upper bound. Diff: {diff}")
            upper = torch.where(middle > upper, middle, upper)

        assert (lower <= middle).all(), "Lower bound must be less than or equal to middle bound."
        assert (middle <= upper).all(), "Middle bound must be less than or equal to upper bound."

        return torch.stack([lower, middle, upper], dim=1).refine_names("N", "bounds", "C", "H", "W")  # type: ignore
    