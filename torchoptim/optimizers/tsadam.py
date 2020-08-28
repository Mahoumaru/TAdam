import math
import torch
from torch.optim.optimizer import Optimizer

class TSAdam(Optimizer):
    r"""Implements a robust version of SAdam algorithm.
    It has been proposed in `SAdam: A Variant of Adam for Strongly Convex Functions`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta1 (float, optional): coefficient used for computing
            running averages of gradient(default: 0.9)
        gamma (float, optional): coefficient used for computing
            running averages of gradients square (default: beta1)
        eps (float, optional): term (vanishing factor) added to the denominator to improve
            numerical stability. It has been named delta in the SAdam paper. (default: 1e-2)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        xis (Tuple[float, float], optional): coefficients used in the decay scheme of eps (or delta).
            First proposed in 'Variants of RMSProp and Adagrad with Logarithmic Regret Bounds'. (default: (0.1, 1.))
    .. _SAdam: A Variant of Adam for Strongly Convex Functions:
        https://arxiv.org/abs/1905.02957
    .. _Variants of RMSProp and Adagrad with Logarithmic Regret Bounds:
        https://arxiv.org/abs/1706.05507
    """

    def __init__(self, params, lr=1e-3, k_dof=1.0, beta1=0.9, gamma=None, eps=1e-2,
                 xis=(0.1, 1.), weight_decay=0, vary_eps=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps value: {}".format(eps))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta1))
        if gamma is not None:
            if not 0.0 <= gamma <= 1.0:
                raise ValueError("Invalid gamma parameter: {}".format(gamma))
        else:
            gamma = beta1
        if not 0.0 <= xis[0]:
            raise ValueError("Invalid xi parameter at index 0: {}".format(xis[0]))
        if not 0.0 <= xis[1]:
            raise ValueError("Invalid xi parameter at index 1: {}".format(xis[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not (0.0 < k_dof or math.inf == k_dof):
            raise ValueError("Invalid degrees of freedom scale factor: {}".format(k_dof))
        defaults = dict(lr=lr, k_dof=k_dof, beta1=beta1, gamma=gamma, xis=xis, eps=eps,
                        weight_decay=weight_decay, vary_eps=vary_eps)
        super(TSAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(TSAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('vary_eps', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('TSAdam does not support sparse gradients, please consider SparseAdam instead')
                vary_eps = group['vary_eps']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Definition of weight W_t
                    beta1 = group['beta1']
                    state['W_t'] = torch.tensor(0.) + beta1 / (1.0 - beta1)
                    # Dimension d of the parameters
                    state['dim'] = float(p.numel())
                    # Degrees of freedom, initialized to the parameters dimension or to the user specified value
                    if not group["k_dof"] == math.inf:
                        state['dof'] = torch.tensor(0.) + group["k_dof"] * state['dim']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                Wt = state['W_t']
                if vary_eps:
                    xi1, xi2 = group['xis']
                beta1, gamma = group['beta1'], group['gamma']

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
                    Wt.mul_(2.0 - 1.0 / beta1).add_(wt)
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(betaw).add_(grad, alpha=1 - betaw)
                beta2 = 1 - (gamma / state['step'])
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if vary_eps:
                    delta = xi2 * torch.exp(-xi1 * exp_avg_sq)
                else:
                    delta = group['eps']
                denom = (exp_avg_sq / bias_correction2).mul_(state['step']).add_(delta)

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
