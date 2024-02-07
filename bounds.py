# coding=utf-8
# Copyright 2019 The Interval Bound Propagation Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Definition of input bounds to each layer."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class IntervalBounds():
    """Axis-aligned bounding box."""

    def __init__(self, lower, upper):
        super(IntervalBounds, self).__init__()

        assert isinstance(lower, torch.Tensor)
        assert isinstance(upper, torch.Tensor)

        self._lower = lower
        self._upper = upper

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @property
    def shape(self):
        return self.lower.shape

    def __iter__(self):
        yield self.lower
        yield self.upper

    @classmethod
    def convert(cls, bounds):
        if isinstance(bounds, torch.Tensor):
            return cls(bounds, bounds)
        bounds = bounds.concretize()
        if not isinstance(bounds, cls):
            raise ValueError(f'Cannot convert {bounds} to {cls.__name__}')
    
        return bounds

    def apply_linear(self, wrapper, w, b):
        return self._affine(w, b, F.linear)

    def apply_conv1d(self, wrapper, w, b, padding, stride):
        return self._affine(w, b, F.conv1d, padding=padding, stride=stride)

    def apply_conv2d(self, wrapper, w, b, padding, strides):
        return self._affine(w, b, F.conv2d,
                        padding=padding, strides=strides)

    def _affine(self, w, b, fn, **kwargs):
        c = (self.lower + self.upper) / 2.
        r = (self.upper - self.lower) / 2.
        c = fn(c, w, **kwargs)
        if b is not None:
            c = c + b
        r = fn(r, torch.abs(w), **kwargs)
        return IntervalBounds(c - r, c + r)

    def apply_increasing_monotonic_fn(self, wrapper, fn, *args, **parameters):
        args_lower = [self.lower] + [a.lower for a in args]
        args_upper = [self.upper] + [a.upper for a in args]
        
        return IntervalBounds(fn(*args_lower), fn(*args_upper))

    def apply_piecewise_monotonic_fn(self, wrapper, fn, boundaries, *args):
        valid_values = []
        for a in [self] + list(args):
            vs = []
            vs.append(a.lower)
            vs.append(a.upper)
        for b in boundaries:
            vs.append(
                torch.maximum(a.lower, torch.minimum(a.upper, b * torch.ones_like(a.lower))))
        valid_values.append(vs)
        outputs = []
        for inputs in itertools.product(*valid_values):
            outputs.append(fn(*inputs))
        outputs = torch.stack(outputs, dim=-1)
        
        return IntervalBounds(torch.min(outputs, dim=-1),
                            torch.max(outputs, dim=-1))

    def apply_batch_norm(self, wrapper, mean, variance, scale, bias, epsilon):

        # Element-wise multiplier
        multiplier = torch.rsqrt(variance + epsilon)

        if scale is not None:
            multiplier *= scale
        w = multiplier

        # Element-wise bias
        b = -multiplier * mean

        if bias is not None:
            b += bias

        b = b.squeeze(0)

        # Because the scale might be negative, we need to apply a strategy similar
        # to linear
        c = (self.lower + self.upper) / 2.
        r = (self.upper - self.lower) / 2.
        
        c = F.linear(c, weight=w, bias=b)
        r = F.linear(r, weight=w.abs(), bias=torch.zeros_like(r))

        return IntervalBounds(c - r, c + r)

    def apply_batch_reshape(self, wrapper, shape):

        return IntervalBounds(self.lower.view(shape),
                              self.upper.view(shape))

    def apply_softmax(self, wrapper):
        ub = self.upper
        lb = self.lower

        # Keep diagonal and take opposite bound for non-diagonals

        lbs = torch.diagonal(lb) + ub.unsqueeze(dim=-2) - torch.diagonal(ub)
        ubs = torch.diagonal(ub) + lb.unsqueeze(dim=-2) - torch.diagonal(lb)

        # Get diagonal entries after softmax operation.
        ubs = torch.stack(tuple(t.diag() for t in torch.unbind(ubs,0)))
        lbs = torch.stack(tuple(t.diag() for t in torch.unbind(lbs,0)))
        return IntervalBounds(lbs, ubs)
