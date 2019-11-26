from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torchvision.models as models


class _netVGGFeatures(nn.Module):
    def __init__(self):
        super(_netVGGFeatures, self).__init__()
        self.vggnet = models.vgg16(pretrained=True).cuda()
        self.layer_ids = [2, 7, 12, 21, 30]

    def main(self, z, levels):
        layer_ids = self.layer_ids[:levels]
        id_max = layer_ids[-1] + 1
        output = []
        for i in range(id_max):
            z = self.vggnet.features[i](z)
            if i in layer_ids:
                output.append(z)
        return output

    def forward(self, z, levels):
        output = self.main(z, levels)
        return output


class _VGGDistance(nn.Module):
    def __init__(self, levels):
        super(_VGGDistance, self).__init__()
        self.vgg = _netVGGFeatures()
        self.levels = levels

    def forward(self, I1, I2):
        b_sz = I1.size(0)
        f1 = self.vgg(I1, self.levels)
        f2 = self.vgg(I2, self.levels)
        loss = torch.abs(I1 - I2).view(b_sz, -1).mean(1)
        for i in range(self.levels):
            layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
            loss = loss + layer_loss
        return loss

class _VGGFixedDistance(nn.Module):
    def __init__(self):
        super(_VGGFixedDistance, self).__init__()
        self.vgg = _netVGGFeatures()
        self.up = nn.UpsamplingBilinear2d(size=(224, 224))

    def forward(self, I1, I2):
        b_sz = I1.size(0)
        f1 = self.vgg(self.up(I1), 5)
        f2 = self.vgg(self.up(I2), 5)
        loss = 0 # torch.abs(I1 - I2).view(b_sz, -1).mean(1)
        for i in range(5):
            if i < 4:
                continue
            layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
            loss = loss + layer_loss
        return loss


class _VGGMSDistance(nn.Module):
    def __init__(self):
        super(_VGGMSDistance, self).__init__()
        self.vgg = _netVGGFeatures()
        self.subs = nn.AvgPool2d(4)

    def forward(self, I1, I2):
        f1 = self.vgg(I1, 5)
        f2 = self.vgg(I2, 5)
        loss = torch.abs(I1 - I2).mean()
        for i in range(5):
            layer_loss = torch.abs(f1[i] - f2[i]).mean()
            # .mean(3).mean(2).mean(0).sum()
            loss = loss + layer_loss

        f1 = self.vgg(self.subs(I1), 4)
        f2 = self.vgg(self.subs(I2), 4)
        loss = loss + torch.abs(self.subs(I1) - self.subs(I2)).mean()
        for i in range(4):
            layer_loss = torch.abs(f1[i] - f2[i]).mean()
            # .mean(3).mean(2).mean(0).sum()
            loss = loss + layer_loss

        return loss
