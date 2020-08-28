# -*- coding: utf-8 -*-
import math
import torch
from torch.optim import Optimizer

#########################################################################################################
class RoAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.999),
                 eps=1e-8, weight_decay=0, amsgrad=False, bounds=(0.1, 10.)):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= amsgrad <= 1.0:
            raise ValueError("Invalid amsgrad parameter: {}".format(amsgrad))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, bounds=bounds)
        super(RoAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RoAdam, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, floss, closure=None):
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
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['prev_loss'] = torch.zeros_like(p, memory_format=torch.preserve_format).sum().add_(1.)#torch.zeros(1).add_(1)
                    # Exponential moving average of the relative loss (prediction) error
                    state['exp_avg_d'] = torch.zeros_like(p, memory_format=torch.preserve_format).sum().add_(1.)#torch.zeros(1).add_(1)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                prev_loss, exp_avg_d = state['prev_loss'], state['exp_avg_d']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq'].mul_(amsgrad)
                beta1, beta2, beta3 = group['betas']
                k, K = group['bounds']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                l_ratio = floss.abs().div(prev_loss.abs())
                if floss.abs() >= prev_loss.abs():
                    r = l_ratio.clamp(min=k, max=K)#torch.min(torch.max(k, l_ratio), K)
                else:
                    r = l_ratio.clamp(min=1/K, max=1/k)#torch.min(torch.max(torch.tensor(1/K), l_ratio), torch.tensor(1/k))
                exp_avg_d.mul_(beta3).add_(r, alpha=1 - beta3)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).mul_(exp_avg_d).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                prev_loss.mul_(0).add_(floss.abs())

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
