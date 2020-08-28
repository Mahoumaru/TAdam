import math
import torch
from torch.optim import Optimizer

################################## TYogi #################################################
class TYogi(Optimizer):
    r"""Implements a Robust version of Yogi.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        k_dof (float or inf, optional): the degrees of freedom of the student-t
            distribution nu is defined as k_dof * dimension of the param_groups
            (default: 1.0)
    .. _Adaptive Methods for Nonconvex Optimization (Yogi):
        https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=1e-3, k_dof=1.0, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0):
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
        defaults = dict(lr=lr, k_dof=k_dof, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(TYogi, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TYogi, self).__setstate__(state)

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
                    raise RuntimeError('TYogi, just as Adam, does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
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
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

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
                grad_squared = grad.pow(2)
                exp_avg_sq.addcmul_(torch.sign(exp_avg_sq - grad_squared), grad_squared, value=-(1 - beta2))

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
