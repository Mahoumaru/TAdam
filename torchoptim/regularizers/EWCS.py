# coding:utf-8

import math
import torch
from torch.optim import Optimizer

################################## EWCS #################################################
class EWCS(Optimizer):
    r"""Implements EWCS algorithm.

    It has been proposed in `hoge`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lsp (float, optional): regularization coeff for sparsity (default: 1e-7)
        per (float, optional): consolidation percentage (default: 0.75)
        nums (Tuple[float, float, float], optional): the numbers of maximum samples for moving averages (default: (1e+6, 1e+4, 1e+6))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-16)

    """

    def __init__(self, params, lsp=1e-7, per=0.75, nums=(1e+6, 1e+4, 1e+6), eps=1e-16):
        if not 0.0 <= lsp:
            raise ValueError("Invalid sparse coefficient: {}".format(lsp))
        if not 0.0 <= per <= 1.0:
            raise ValueError("Invalid consolidation rate: {}".format(per))
        if not 0.0 <= nums[0]:
            raise ValueError("Invalid number of maximum samples for p1: {}".format(nums[0]))
        if not 0.0 <= nums[1]:
            raise ValueError("Invalid number of maximum samples for p2: {}".format(nums[1]))
        if not 0.0 <= nums[2]:
            raise ValueError("Invalid number of maximum samples for loss: {}".format(nums[2]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        defaults = dict(lsp=lsp, per=per, nums=nums, eps=eps)
        super(EWCS, self).__init__(params, defaults)

    # def __setstate__(self, state):
    #     super(EWCS, self).__setstate__(state)

    def step(self, loss, nb=1):
        """Performs a single optimization step.

        Arguments:
            loss (float): loss used for calculating gradient.
            nb (int, optional): minibatch size.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('EWCS does not support sparse gradients')

                state = self.state[p]
                w_ = p.data
                F_ = p.grad.data.pow(2)
                F_.sub_(group['lsp']).clamp_(min=0.0)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # parameters for p(theta | t_old) =: p1
                    state['mu1'] = torch.zeros_like(w_)
                    state['tau1'] = torch.zeros_like(w_) + group['eps']
                    # parameters for p(theta | t_new) =: p2
                    state['mu2'] = w_.clone()
                    state['tau2'] = F_.clone()
                    # mean and variance of loss as normal distribution
                    state['lmean'] = 0.0
                    state['lvar'] = math.pow(3.0 * loss, 2)
                    # topk to consolidate
                    state['topk'] = max([1, math.floor((1.0 - group['per']) * w_.numel())])

                beta1, beta2, betal = group['nums']
                beta1 = max(0.0, 1.0 - nb / beta1)
                beta2 = max(0.0, 1.0 - nb / beta2)
                betal = max(0.0, 1.0 - nb / betal)

                state['step'] += 1

                # update mean and variance of loss
                state['lvar'] = betal * state['lvar'] + betal * (1.0 - betal) * math.pow(loss - state['lmean'], 2)
                state['lmean'] = betal * state['lmean'] + (1.0 - betal) * loss
                # get normalized loss through sigmoid
                lnorm = 0.5 * (1.0 + math.tanh(2.0 * (loss - state['lmean']) / (math.sqrt(state['lvar']) + group['eps'])))

                # update p1: when large loss, p2 is added to p1
                beta1 = (1.0 - lnorm) * (1.0 - beta1) + beta1
                state['mu1'].mul_(state['tau1']).add_((1.0 - beta1), state['tau2'].mul(state['mu2'])).div_(state['tau1'].add((1.0 - beta1), state['tau2']).add_(group['eps']))
                state['tau1'].add_((1.0 - beta1), state['tau2'])

                # update p2: when small loss, p2 is changed to p3 while losing its info
                beta2 = lnorm * (1.0 - beta2) + beta2
                state['mu2'].mul_(beta1 * beta2).add_((1.0 - beta2), w_)
                state['tau2'].mul_(beta1 * beta2).add_((1.0 - beta2), F_)

                # get total regularization coefficients
                importance, _ = state['tau1'].flatten().kthvalue(state['topk'])
                coeff = state['tau1'].div(importance.add_(group['eps']))

                # regularize parameter toward mean or zero
                ewc = w_.sub(state['mu1']).mul_(coeff.clamp(max=1.0))
                l1 = w_.abs().min(coeff.pow(-1).mul_(group['lsp'])).mul_(w_.sign())
                w_.sub_(ewc.add(l1))

        return loss
