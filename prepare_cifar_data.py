import sys
import os
import pickle
import numpy as np


train_list = ['data_batch_1', 'data_batch_2',
              'data_batch_3', 'data_batch_4', 'data_batch_5']

test_list = ['test_batch']

root = "../../../datasets/cifar-10-batches-py/"
train_ims = []
train_l = []
for fentry in train_list:
  file = os.path.join(root, fentry)
  fo = open(file, 'rb')
  entry = pickle.load(fo, encoding='latin1')
  train_ims.append(entry['data'])
  train_l += entry['labels']
  fo.close()
train_ims = np.concatenate(train_ims)
train_ims = train_ims.reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))
np.save('train_cifar.npy', train_ims)


file = os.path.join(root, test_list[0])
fo = open(file, 'rb')
entry = pickle.load(fo, encoding='latin1')
test_ims = entry['data']
test_l = entry['labels']
fo.close()
test_ims = test_ims.reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
np.save('test_cifar.npy', test_ims)

