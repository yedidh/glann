from mnist import MNIST
import numpy as np

mndata = MNIST('../../../datasets/fashion/')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
train_images = np.array(train_images)
test_images = np.array(test_images)
np.save('train_fashion.npy', train_images.reshape((-1, 28, 28, 1)))
np.save('test_fashion.npy', test_images.reshape((-1, 28, 28, 1)))
