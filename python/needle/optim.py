"""Optimization module"""

import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data + self.weight_decay * p.data
            u_prev = self.u.get(p, 0)
            u_curr = self.momentum * u_prev + (1 - self.momentum) * grad
            self.u[p] = u_curr.detach()
            p.data -= self.lr * u_curr
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data + self.weight_decay * p.data
            m_prev = self.m.get(p, 0)
            v_prev = self.v.get(p, 0)

            m_curr = self.beta1 * m_prev + (1 - self.beta1) * grad
            v_curr = self.beta2 * v_prev + (1 - self.beta2) * grad**2
            self.m[p] = m_curr.detach()
            self.v[p] = v_curr.detach()

            m_hat = m_curr / (1 - self.beta1**self.t)
            v_hat = v_curr / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)
        ### END YOUR SOLUTION
