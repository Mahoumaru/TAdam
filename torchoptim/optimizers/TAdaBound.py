# -*- coding: utf-8 -*-
import math
import torch
import numpy as np
from torch.optim import Optimizer

#########################################################################################################
class TAdaBound(Optimizer):
    r"""Implements a Robust version of AdaBound.

    .. _AdaBound: Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
          https://openreview.net/pdf?id=Bkg3g2R9FX
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
        super(TAdaBound, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TAdaBound, self).__setstate__(state)
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
                    raise RuntimeError('t-Adam, just as Adam, does not support sparse gradients, please consider SparseAdam instead')
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
                    # Definition of weight W_t
                    beta1, beta2 = group['betas']
                    state['W_t'] = torch.zeros(1) + beta1 / (1.0 - beta1)
                    # Definition of weight w_t
                    state['w_t'] = torch.zeros(1)
                    # Dimension d of the parameters
                    state['dim'] = 1
                    for i in p.data.size():
                        state['dim']*=i
                    # Degrees of freedom, initialized to the parameters dimension
                    state['dof'] = torch.zeros_like(p.data).add(1).sum()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                Wt, wt = state['W_t'], state['w_t']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                dof = state['dof']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Weights computation
                wt.mul_(0.0).add_(grad.sub(exp_avg).pow_(2).div_(exp_avg_sq.add(group['eps'])).sum())
                wt.add_(dof).pow_(-1).mul_((state['dim'] + dof))
                betaw = Wt.div(Wt.add(wt)).item()
                Wt.mul_(2.0 - 1.0/beta1).add_(wt)
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(betaw).add_((1 - betaw), grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # Compute the upper and the lower bound:
                eta_low = (1 - (1 / ((1 - beta2) ** (state['step'] + 1) + group['eps']))) * group['lr']
                eta_up = (1 + (1 / ((1 - beta2) ** (state['step']) + group['eps']))) * group['lr']
                # TAMSBound
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] / denom
                step_size = np.clip(step_size.cpu().numpy(), a_max = eta_up, a_min = eta_low)
                #step_size.div_(math.sqrt(state['step']))
                step_size = torch.tensor(step_size).to(grad.device)
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.add_(-step_size * exp_avg)

        return loss
