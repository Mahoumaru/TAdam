# -*- coding: utf-8 -*-
import math
import torch
from torch.optim import Optimizer

#########################################################################################################
class TRAdam(Optimizer):
    r"""Implements a Robust version of RAdam (Rectified Adam).

    .. _RAdam: On the variance of the adaptive learning rate and beyond:
        https://arxiv.org/pdf/1908.03265.pdf
    """

    def __init__(self, params, lr=1e-3, k_dof=1.0, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not (0.0 <= k_dof or math.inf == k_dof):
            raise ValueError("Invalid degrees of freedom scale factor: {}".format(k_dof))
        if not 0.0 <= amsgrad <= 1.0:
            raise ValueError("Invalid amsgrad parameter: {}".format(amsgrad))
        defaults = dict(lr=lr, k_dof=k_dof, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(TRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TRAdam, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('amsgrad', False)

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
                    state['W_t'] = torch.tensor(0.) + beta1 / (1.0 - beta1)
                    # Dimension d of the parameters
                    state['dim'] = p.data.numel()
                    # Degrees of freedom, initialized to the parameters dimension
                    if not group["k_dof"] == math.inf:
                        state['dof'] = torch.tensor(0.) + group["k_dof"] * state['dim']
                    # Maximum length of the approximated SMA
                    state['rho_inf'] = ((2. / (1. - group['betas'][1])) - 1.)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                Wt = state['W_t']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq'].mul_(amsgrad)
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Weights computation
                if group["k_dof"] == math.inf:
                    betaw = beta1
                else:
                    wt = grad.sub(exp_avg).pow_(2).div_(exp_avg_sq.add(group['eps'])).sum()
                    wt.add_(state['dof']).pow_(-1).mul_(state['dim'] + state['dof'])
                    betaw = Wt.div(Wt.add(wt))
                    Wt.mul_(2.0 - 1.0/beta1).add_(wt)
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(betaw).add_(grad.mul(1 - betaw))
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
