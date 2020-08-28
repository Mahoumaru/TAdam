import math
import torch
from torch.optim.optimizer import Optimizer


__all__ = ('TDiffGrad',)


class TDiffGrad(Optimizer):
    r"""Implements a robust version of the DiffGrad algorithm.
    DiffGrad has been proposed in `DiffGrad: An Optimization Method for
    Convolutional Neural Networks`__.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.DiffGrad(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ https://arxiv.org/abs/1909.11015
    Note:
        Reference code: https://github.com/shivram1987/diffGrad
    """

    def __init__(self, params, lr = 1e-3, k_dof=1.0, betas = (0.9, 0.999),
                 eps = 1e-8, weight_decay = 0.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not (0.0 < k_dof or math.inf == k_dof):
            raise ValueError("Invalid degrees of freedom scale factor: {}".format(k_dof))
        defaults = dict(lr=lr, k_dof=k_dof, betas=betas, eps=eps, weight_decay=weight_decay)
        super(TDiffGrad, self).__init__(params, defaults)

    def step(self, closure=None):
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('TDiffGrad, just as Adam, does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous gradient
                    state['previous_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Definition of weight W_t
                    state['W_t'] = torch.tensor(0.) + beta1 / (1.0 - beta1)
                    # Dimension d of the parameters
                    state['dim'] = float(p.numel())
                    # Degrees of freedom, initialized to the parameters dimension or to the user specified value
                    if not group["k_dof"] == math.inf:
                        state['dof'] = torch.tensor(0.) + group["k_dof"] * state['dim']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                previous_grad = state['previous_grad']
                Wt = state['W_t']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p, alpha=group['weight_decay'])

                # Weights computation
                if group["k_dof"] == math.inf:
                    betaw = beta1
                else:
                    wt = grad.sub(exp_avg).pow_(2).div_(exp_avg_sq.add(group['eps'])).sum()
                    wt.add_(state['dof']).pow_(-1).mul_(state['dim'] + state['dof'])
                    betaw = Wt.div(Wt.add(wt))
                    Wt.mul_(2.0 - 1.0 / beta1).add_(wt)
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(betaw).add_(grad, alpha=1 - betaw)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # compute diffgrad coefficient (dfc)
                diff = torch.abs(previous_grad - grad)
                dfc = torch.div(1.0, (1.0 + torch.exp(-diff)))
                state['previous_grad'] = grad.clone()

                # update momentum with dfc
                exp_avg1 = exp_avg * dfc
                step_size = (group['lr'] * math.sqrt(bias_correction2) / bias_correction1)
                p.data.addcdiv_(exp_avg1, denom, value=-step_size)

        return loss
