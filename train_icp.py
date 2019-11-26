import numpy as np
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import icp
import torchvision.utils as vutils
import model


# Arguments
parser = argparse.ArgumentParser(
    description='Train ICP.'
)
parser.add_argument('config', type=str, help='Path to cofig file.')

args = parser.parse_args()

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

with open(args.config, 'r') as f:
  params = yaml.load(f, Loader=Loader)

rn = params['name']
nc = params['nc']
sz = params['sz']
nz = params['glo']['nz']
do_bn = params['glo']['do_bn']
dim = params['icp']['dim']
nepoch = params['icp']['total_epoch']

W = torch.load('runs/nets_%s/netZ_nag.pth' % (rn))
W = W['emb.weight'].data.cpu().numpy()

netG = model._netG(nz, sz, nc, do_bn).cuda()
state_dict = torch.load('runs/nets_%s/netG_nag.pth' % (rn))
netG.load_state_dict(state_dict)


icpt = icp.ICPTrainer(W, dim)
icpt.train_icp(nepoch)
torch.save(icpt.icp.netT.state_dict(), 'runs/nets_%s/netT_nag.pth' % rn)

z = icpt.icp.netT(torch.randn(64, dim).cuda())
ims = netG(z)
vutils.save_image(ims,
                  'runs/ims_%s/samples.png' % (rn),
                  normalize=False)
