 This repository is mainly dedicated to t-Adam, i.e. the t-momentum based Adam optimizer, and will no longer be updated.
 
 Please visit the following official t-momentum repository: [here](https://github.com/Mahoumaru/t-momentum.git).
 The t-momentum repository will be updated with the latest advances concerning the t-momentum algorithm in general.

# torchoptim
TAdam (http://arxiv.org/abs/2003.00179) and RoAdam optimizers for neural networks.

Implemented using the pytorch library.

## how to use

1. install by pip
```bash
git clone https://github.com/Mahoumaru/torchoptim.git
cd torchoptim
pip install -e .
```
2. replace optimizer!
```python
from torchoptim.optimizers.TAdam import TAdam
optimizer = TAdam(net.parameters())
```
3. adjust hyperparameters (same as Adam: lr, betas, eps, weight_decay, amsgrad. - N.B. RoAdam has a third beta. -)

## Demos
 Contains the codes used for the results in the paper (http://arxiv.org/abs/2003.00179)

 In order to use the reinforcement learning code, you need to install the rlpyt library, developped by the Berkley A.I. research team: https://github.com/astooke/rlpyt.git
 Follow the instructions on the link to install it.
 
 Note that a modified version of PPO and A2C is included under the reinforcement learning demo folder. The only difference with the BAIR implementation is the fact that this modified version does not use the gradient norm clipping scheme (More details in the paper).
