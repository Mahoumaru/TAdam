# coding:utf-8

import math
import torch
from torch.optim import Optimizer

################################## t-Adam #################################################
class TLaProp(Optimizer):
    r"""Implements a Robust version of Adam.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean or [0,1], optional): to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            if float is given, v_max is decayed (default: False)
        k_dof (float or inf, optional): the degrees of freedom of the student-t
            distribution nu is defined as k_dof * dimension of the param_groups
            (default: 1.0)

    .. _LaProp: a Better Way to Combine Momentum with Adaptive Gradient:
        https://arxiv.org/abs/2002.04839
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
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
        if not (0.0 < k_dof or math.inf == k_dof):
            raise ValueError("Invalid degrees of freedom scale factor: {}".format(k_dof))
        if not 0.0 <= amsgrad <= 1.0:
            raise ValueError("Invalid amsgrad parameter: {}".format(amsgrad))
        defaults = dict(lr=lr, k_dof=k_dof, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(TLaProp, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TLaProp, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('amsgrad', False)

    @torch.no_grad()
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
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('TLaProp, just as Adam, does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Definition of weight W_t
                    beta1, beta2 = group['betas']
                    state['W_t'] = torch.tensor(0.) + beta1 / (1.0 - beta1)
                    # Dimension d of the parameters
                    state['dim'] = float(p.numel())
                    # Degrees of freedom, initialized to the parameters dimension or to the user specified value
                    if not group["k_dof"] == math.inf:
                        state['dof'] = torch.tensor(0.) + group["k_dof"] * state['dim']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                Wt = state['W_t']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq'].mul_(amsgrad)
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # second-order momentum
                exp_avg_sq.mul_(beta2).add_(grad.pow(2).mul_(1 - beta2))
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # first-order momentum
                gv = grad.div(denom)
                if group["k_dof"] == math.inf:
                    betaw = beta1
                else:
                    wt = gv.sub(exp_avg).pow_(2).sum()
                    wt.add_(state['dof']).pow_(-1).mul_(state['dim'] + state['dof'])
                    betaw = Wt.div(Wt.add(wt))
                    Wt.mul_(2.0 - 1.0/beta1).add_(wt)
                exp_avg.mul_(betaw).add_(gv.mul(1 - betaw))

                step_size = group['lr'] / bias_correction1

                p.add_(exp_avg, alpha=-step_size)

        return loss
