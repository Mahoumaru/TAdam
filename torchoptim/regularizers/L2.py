# coding:utf-8

import math
import torch
from torch.optim import Optimizer

################################## L2 #################################################
class L2(Optimizer):
    r"""Implements L2 algorithm.

    It has been proposed in `hoge`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): regularization rate (default: 1e-10)

    """

    def __init__(self, params, lr=1e-5):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(L2, self).__init__(params, defaults)

    # def __setstate__(self, state):
    #     super(L2, self).__setstate__(state)

    def step(self, loss, nb=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('L2 does not support sparse gradients')

                state = self.state[p]
                w_ = p.data

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                state['step'] += 1

                # regularize parameter toward zero
                w_.sub_(group['lr'], w_)

        return loss
