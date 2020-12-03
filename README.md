# MS-SSIM_L1_LOSS
Pytorch implementation of MS-SSIM L1 Loss function for image restoration.

# How to use
import this .py file into your project.
```
from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
criterion = MS_SSIM_L1_LOSS()

# your pytorch tensor x, y with [B, C, H, W] dimension on cuda device 0
loss = criterion(x, y)
```
Please check demo.py for more details.

# Requirements:
pytorch (only tested on pytorch 1.7.0 on windows 10 x64, other versions should work.)
## Optional (only for testing this loss function)
numpy
PIL

# References:
# [1] H. Zhao, O. Gallo, I. Frosio and J. Kautz, "Loss Functions for Image Restoration With Neural Networks," in IEEE Transactions on Computational Imaging, vol. 3, no. 1, pp. 47-57, March 2017
# [2] https://github.com/NVlabs/PL4NN
# [3] https://github.com/VainF/pytorch-msssim
