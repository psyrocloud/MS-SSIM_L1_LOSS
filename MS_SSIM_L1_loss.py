# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 00:28:15 2020

@author: Yunpeng Li, Tianjin University
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class MS_SSIM_L1_LOSS(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0,
                 cuda_dev=0,):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation=compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3*idx+0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=3, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=3, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=3, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=3, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=3, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=3, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation*loss_mix
#        first = loss_ms_ssim.mean().cpu().numpy()*self.compensation
#        second = (gaussian_l1/self.DR).mean().cpu().numpy()*self.compensation
#        return loss_mix.mean(), (first, first*self.alpha), (second, second*self.alpha)
        return loss_mix.mean()


def pil2tensor(im):  # in: [PIL Image with 3 channels]. out: [B=1, C=3, H, W] (0, 1)
    return torch.Tensor((np.float32(im) / 255).transpose(2, 0 ,1)).unsqueeze(0)


if __name__ == '__main__':
    '''Test of this loss function'''
    # load image
    from PIL import Image, ImageFilter
    import numpy as np
    im = Image.open('lena.png')
    gt = pil2tensor(im).cuda(0)
    blur_levels = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    noise_levels = [5, 10 ,20, 40, 80, 160]
    im1 = np.float32(im)
    imgs = []
    imgs2 = []

    for bl in blur_levels:
        imgs.append(im.filter(ImageFilter.GaussianBlur(radius = bl)))
    for nl in noise_levels:
        nr = np.float32(Image.effect_noise((512, 512), nl)) - 128
        ng = np.float32(Image.effect_noise((512, 512), nl)) - 128
        nb = np.float32(Image.effect_noise((512, 512), nl)) - 128
        noise = np.stack((nr, ng, nb), 2)
        imn = Image.fromarray(np.uint8(np.clip(im1 + noise, 0, 255)))
        imgs2.append(imn)

    # test the loss
    LOSS = MS_SSIM_L1_LOSS()
    loss_blur = []
    loss_noise = []
    for img in imgs:
        img = pil2tensor(img).cuda(0)
        loss = LOSS(img, gt)
        loss_blur.append(loss)
    for img in imgs2:
        img = pil2tensor(img).cuda(0)
        loss = LOSS(img, gt)
        loss_noise.append(loss)

