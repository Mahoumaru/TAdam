# -*- coding: utf-8 -*-
import math
import torch
from torch.optim import Optimizer

#########################################################################################################
class TAdaBound(Optimizer):
    r"""Implements a Robust version of AdaBound.

    .. _AdaBound: Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
          https://openreview.net/pdf?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(TAdaBound, self).__init__(params, defaults)
        
        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(TAdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('TAdaBound, just as Adam, does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
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
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss
