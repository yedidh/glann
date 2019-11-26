from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import shutil
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F 
import torchvision.utils as vutils
import model
import utils


class GLO():
    def __init__(self, glo_params, image_params, rn):
        self.netZ = model._netZ(glo_params.nz, image_params.n)
        self.netZ.apply(model.weights_init)
        self.netZ.cuda()
        self.rn = rn

        self.netG = model._netG(glo_params.nz, image_params.sz[0], image_params.nc, glo_params.do_bn)
        self.netG.apply(model.weights_init)
        self.netG.cuda()
        # self.netG = nn.DataParallel(self.netG)

        self.vis_n = 64

        fixed_noise = torch.FloatTensor(self.vis_n,
                                        glo_params.nz).normal_(0, 1)
        self.fixed_noise = fixed_noise.cuda()

        self.glo_params = glo_params
        self.image_params = image_params

        # lap_criterion = pyr.MS_Lap(4, 5).cuda()
        self.dist = utils.distance_metric(image_params.sz[0], image_params.nc,
                                          glo_params.force_l2)
        # self.dist = nn.DataParallel(self.dist)

    def train(self, ims_np, opt_params, vis_epochs=1):
        for epoch in range(opt_params.epochs):
            er = self.train_epoch(ims_np, epoch, opt_params)
            print("NAG Epoch: %d Error: %f" % (epoch, er))
            torch.save(self.netZ.state_dict(), 'runs/nets_%s/netZ_nag.pth' % self.rn)
            torch.save(self.netG.state_dict(), 'runs/nets_%s/netG_nag.pth' % self.rn)
            if epoch % vis_epochs == 0:
                self.visualize(epoch, ims_np)

    def train_epoch(self, ims_np, epoch, opt_params):
        rp = np.random.permutation(self.image_params.n)
        # Compute batch size
        batch_size = opt_params.batch_size
        batch_n = self.image_params.n // batch_size
        # Compute learning rate
        decay_steps = epoch // opt_params.decay_epochs
        lr = opt_params.lr * opt_params.decay_rate ** decay_steps
        # Initialize optimizers
        optimizerG = optim.Adam(self.netG.parameters(), lr=lr * opt_params.factor,
                                betas=(0.5, 0.999))
        optimizerZ = optim.Adam(self.netZ.parameters(), lr=lr,
                                betas=(0.5, 0.999))
        # Start optimizing
        er = 0
        for i in range(batch_n):
            # Put numpy data into tensors
            np_idx = rp[i * batch_size: (i + 1) * batch_size]
            idx = torch.from_numpy(np_idx).long().cuda()
            np_data = ims_np[rp[i * batch_size: (i + 1) * batch_size]]
            images = torch.from_numpy(np_data).float().cuda()
            # Forward pass
            self.netZ.zero_grad()
            self.netG.zero_grad()
            zi = self.netZ(idx)
            Ii = self.netG(zi)
            if self.image_params.nc == 1:
              Ii = Ii.expand((batch_size, 3, Ii.size(2), Ii.size(3)))
              images = images.expand_as(Ii)
            if self.image_params.sz[0] == 28:
              Ii = F.pad(Ii, (2, 2, 2, 2))
              images = F.pad(images, (2, 2, 2, 2))
            rec_loss = self.dist(2 * Ii - 1, 2 * images - 1)
            rec_loss = rec_loss.mean()
            # Backward pass and optimization step
            rec_loss.backward()
            optimizerG.step()
            optimizerZ.step()
            er += rec_loss.item()
        self.netZ.get_norm()
        er = er / batch_n
        return er

    def visualize(self, epoch, ims_np):
        Igen = self.netG(self.fixed_noise)
        z = utils.sample_gaussian(self.netZ.emb.weight.clone().cpu(),
                                  self.vis_n)
        Igauss = self.netG(z)
        idx = torch.from_numpy(np.arange(self.vis_n)).cuda()
        Irec = self.netG(self.netZ(idx))
        Iact = torch.from_numpy(ims_np[:self.vis_n]).cuda()

        epoch = 0
        # Generated images
        vutils.save_image(Igen.data,
                          'runs/ims_%s/generations_epoch_%03d.png' % (self.rn, epoch),
                          normalize=False)
        # Reconstructed images
        vutils.save_image(Irec.data,
                          'runs/ims_%s/reconstructions_epoch_%03d.png' % (self.rn, epoch),
                          normalize=False)
        vutils.save_image(Iact.data,
                          'runs/ims_%s/act.png' % (self.rn),
                          normalize=False)
        vutils.save_image(Igauss.data,
                          'runs/ims_%s/gaussian_epoch_%03d.png' % (self.rn, epoch),
                          normalize=False)

class GLOTrainer():
    def __init__(self, ims_np, glo_params, rn):
        self.ims_np = ims_np
        self.sz = ims_np.shape[2:4]
        self.rn = rn
        self.nc = ims_np.shape[1]
        self.n = ims_np.shape[0]
        self.image_params = utils.ImageParams(sz=self.sz, nc=self.nc, n=self.n)
        self.glo = GLO(glo_params, self.image_params, rn)
        if not os.path.isdir("runs"):
            os.mkdir("runs")
        shutil.rmtree("runs/ims_%s" % self.rn, ignore_errors=True)
        # shutil.rmtree("nets", ignore_errors=True)
        os.mkdir("runs/ims_%s" % self.rn)
        if not os.path.isdir("runs/nets_%s" % self.rn):
            os.mkdir("runs/nets_%s" % self.rn)

    def train_glo(self, opt_params):
        self.glo.train(self.ims_np, opt_params)
