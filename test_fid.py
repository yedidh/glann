import numpy as np
import sys
import os
import argparse
import yaml
import pickle
import torch
import icp
import utils
import model
from fid_score import calculate_fid_given_ims


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
train_path = params['train_path']
test_path = params['test_path']
d = params['icp']['dim']
nz = params['glo']['nz']
do_bn = params['glo']['do_bn']
nc = params['nc']
sz = params['sz']
batch_size = params['fid']['batch_size']
total_n = params['fid']['n_images']

W = torch.load('runs/nets_%s/netZ_nag.pth' % (rn))
W = W['emb.weight'].data.cpu().numpy()

Zs = utils.sample_gaussian(torch.from_numpy(W), total_n)
Zs = Zs.data.cpu().numpy()

state_dict = torch.load('runs/nets_%s/netT_nag.pth' % rn)
netT = icp._netT(d, nz).cuda()
netT.load_state_dict(state_dict)

netG = model._netG(nz, sz, nc, do_bn).cuda()
state_dict = torch.load('runs/nets_%s/netG_nag.pth' % (rn))
netG.load_state_dict(state_dict)

train_ims = np.load(train_path).astype('float')
test_ims = np.load(test_path).astype('float')

rp = np.random.permutation(len(train_ims))[:total_n]
train_ims = train_ims[rp]
rp = np.random.permutation(len(test_ims))[:total_n]
test_ims = test_ims[rp]

batch_n = total_n // batch_size
ims_glann = np.zeros((batch_n * batch_size, nc, sz, sz))
ims_glo = np.zeros((batch_n * batch_size, nc, sz, sz))
ims_reconstruction = np.zeros((batch_n * batch_size, nc, sz, sz))
for i in range(batch_n):
  z = netT(torch.randn(batch_size, d).cuda())
  ims_glann[i * batch_size: i * batch_size + batch_size] = netG(z).cpu().data.numpy()
  z = torch.from_numpy(Zs[np.arange(batch_size) + batch_size * i]).float().cuda()
  ims_glo[i * batch_size: i * batch_size + batch_size] = netG(z).cpu().data.numpy()
  z = torch.from_numpy(W[np.arange(batch_size) + batch_size * i]).float().cuda()
  ims_reconstruction[i * batch_size: i * batch_size + batch_size] = netG(z).cpu().data.numpy()

ims_glann = ims_glann.transpose((0, 2, 3, 1)) * 255.0
ims_glo = ims_glo.transpose((0, 2, 3, 1)) * 255.0
ims_reconstruction = ims_reconstruction.transpose((0, 2, 3, 1)) * 255.0

if nc == 1:
  train_ims = np.tile(train_ims, (1, 1, 1, 3))
  test_ims = np.tile(test_ims, (1, 1, 1, 3))
  ims_glann = np.tile(ims_glann, (1, 1, 1, 3))
  ims_glo = np.tile(ims_glo, (1, 1, 1, 3))
  ims_reconstruction = np.tile(ims_reconstruction, (1, 1, 1, 3))

fid_optimistic = calculate_fid_given_ims([train_ims.copy(), test_ims.copy()], batch_size)
fid_glo = calculate_fid_given_ims([ims_glo.copy(), test_ims.copy()], batch_size)
fid_glann = calculate_fid_given_ims([ims_glann.copy(), test_ims.copy()], batch_size)
fid_reconstruction = calculate_fid_given_ims([ims_reconstruction.copy(), test_ims.copy()], batch_size)
print("FID Opt %.02f Rec: %0.2f GLANN %0.2f GLO: %0.2f" % (fid_optimistic, fid_reconstruction, fid_glann, fid_glo))
