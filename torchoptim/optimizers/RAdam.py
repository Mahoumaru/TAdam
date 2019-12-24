# -*- coding: utf-8 -*-
import math
import torch
from torch.optim import Optimizer

#########################################################################################################
class RAdam(Optimizer):
    r"""Implements RAdam (Rectified Adam) algorithm.

    .. _On the variance of the adaptive learning rate and beyond:
        https://arxiv.org/pdf/1908.03265.pdf
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
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
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients, please consider SparseRAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    # Maximum length of the approximated SMA
                    state['rho_inf'] = ((2. / (1. - group['betas'][1])) - 1.)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                #exp_avg.mul_(beta1).add_((1 - beta1)*(1 - beta1 ** math.log(state['step'])), grad)
                exp_avg.mul_(beta1).add_((1 - beta1), grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # Compute length of the approximated SMA
                rho_t = state['rho_inf'] - 2 * state['step'] * (beta2 ** state['step']) / (1 - beta2 ** state['step'])
                # If the variance is tractable:
                if rho_t > 4:
                    if amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                    else:
                        denom = exp_avg_sq.sqrt().add_(group['eps'])
                    var_rect = math.sqrt(((rho_t - 4) * (rho_t - 2) * state['rho_inf']) / ((state['rho_inf'] - 4) * (state['rho_inf'] - 2) * rho_t))
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * var_rect * math.sqrt(bias_correction2) / bias_correction1
                else:
                    # else
                    denom = torch.zeros_like(exp_avg).sum().add(1)
                    bias_correction1 = 1 - beta1 ** state['step']
                    step_size = group['lr'] * 1. / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
