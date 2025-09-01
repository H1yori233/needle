from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_keepdims = array_api.max(Z, axis=1, keepdims=True)
        max_z = array_api.max(Z, axis=1)
        exp_z = array_api.exp(Z - max_z_keepdims)
        sum_exp_z = array_api.sum(exp_z, axis=1)
        log_sum_exp = array_api.log(sum_exp_z)

        return Z - (log_sum_exp.reshape((-1, 1)) + max_z_keepdims)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad - exp(node) * out_grad.sum(axes=1).reshape((-1, 1))
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_keepdims = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z = array_api.max(Z, axis=self.axes)
        exp_z = array_api.exp(Z - max_z_keepdims)
        sum_exp_z = array_api.sum(exp_z, axis=self.axes)
        log_sum_exp = array_api.log(sum_exp_z)

        return log_sum_exp + max_z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]

        new_shape = list(Z.shape)
        axes_to_reshape = self.axes if self.axes is not None else range(len(Z.shape))
        if isinstance(axes_to_reshape, int):
            axes_to_reshape = [axes_to_reshape]
        for axis in axes_to_reshape:
            new_shape[axis] = 1

        softmax = exp(Z - node.reshape(new_shape))
        return out_grad.reshape(new_shape) * softmax
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
