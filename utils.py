from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import sys
import numpy as np
import torch
import torch.nn as nn
import vgg_metric


GLOParams = collections.namedtuple('GLOParams', 'nz ngf do_bn mu sd force_l2')
GLOParams.__new__.__defaults__ = (None, None, None, None, None, None)
OptParams = collections.namedtuple('OptParams', 'lr factor ' +
                                                'batch_size epochs ' +
                                                'decay_epochs decay_rate lr_ratio')
OptParams.__new__.__defaults__ = (None, None, None, None, None, None)
ImageParams = collections.namedtuple('ImageParams', 'sz nc n mu sd')
ImageParams.__new__.__defaults__ = (None, None, None)


def distance_metric(sz, nc, force_l2=False):
    # return vgg_metric._VGGFixedDistance()
    if force_l2:
        return nn.L1Loss().cuda()
    if sz == 16:
        return vgg_metric._VGGDistance(2)
    elif sz == 32 or sz == 28:
        return vgg_metric._VGGDistance(3)
    elif sz == 64:
        return vgg_metric._VGGDistance(4)
    elif sz > 64:
        return vgg_metric._VGGMSDistance()


def sample_gaussian(x, m):
    x = x.data.numpy()
    mu = x.mean(0).squeeze()
    cov2 = np.cov(x, rowvar=0)
    z = np.random.multivariate_normal(mu, cov2, size=m)
    z_t = torch.from_numpy(z).float()
    radius = z_t.norm(2, 1).unsqueeze(1).expand_as(z_t)
    z_t = z_t / radius
    return z_t.cuda()


def unnorm(ims, mu, sd):
    for i in range(len(mu)):
        ims[:, i] = ims[:, i] * sd[i]
        ims[:, i] = ims[:, i] + mu[i]
    return ims


def format_im(ims_gen, mu, sd):
    if ims_gen.size(1) == 3:
        rev_idx = torch.LongTensor([2, 1, 0]).cuda()
    elif ims_gen.size(1) == 1:
        rev_idx = torch.LongTensor([0]).cuda()
    else:
        arr = [i for i in range(ims_gen.size(1))]
        rev_idx = torch.LongTensor(arr).cuda()
    # Generated images
    ims_gen = unnorm(ims_gen, mu, sd)
    ims_gen = ims_gen.data.index_select(1, rev_idx)
    ims_gen = torch.clamp(ims_gen, 0, 1)
    return ims_gen
