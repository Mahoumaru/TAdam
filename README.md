# t-Adam
t-Adam: A Robust Stochastic Gradient Optimizer based on Adam

Implemented using the pytorch library.

## how to use

1. install by pip
```bash
git clone https://github.com/kbys-t/torchoptim.git
cd torchoptim
pip install -e .
```
2. replace optimizer!
```python
from torchoptim.TAdam import TAdam
optimizer = TAdam(net.parameters())
```
3. adjust hyperparameters (same as Adam: lr, betas, eps, weight_decay, amsgrad)
