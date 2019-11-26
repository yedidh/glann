import numpy as np
import argparse
import yaml
import torch
import glo
import utils
import os

# Arguments
parser = argparse.ArgumentParser(
    description='Train GLANN.'
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
decay = params['glo']['decay']
total_epoch = params['glo']['total_epoch']
lr = params['glo']['learning_rate']
factor = params['glo']['factor']
nz = params['glo']['nz']
batch_size = params['glo']['batch_size']
do_bn = params['glo']['do_bn']

x = np.load(train_path)
x = x.transpose((0, 3, 1, 2)) / 255.0

glo_params = utils.GLOParams(nz=nz, do_bn=do_bn, force_l2=False)
glo_opt_params = utils.OptParams(lr=lr, factor=factor, batch_size=batch_size, epochs=total_epoch,
                                 decay_epochs=decay, decay_rate=0.5)
nt = glo.GLOTrainer(x, glo_params, rn)
nt.train_glo(glo_opt_params)
