# coding=gbk
import numpy as np
from mlxtend.data import loadlocal_mnist
from train import DataLoader
from train import mlp
import os

X_test, y_test = loadlocal_mnist(
    images_path='/Users/wangjing/Desktop/hw1/data/t10k-images.idx3-ubyte',
    labels_path='/Users/wangjing/Desktop/hw1/data/t10k-labels.idx1-ubyte')


def normalize(x):
    miu = np.mean(x, axis=0, keepdims=True)
    var = np.var(x, axis=0, keepdims=True)
    return (x - miu) / np.sqrt(var + 1e-05)


X_test = normalize(X_test)

num_input = 28 * 28
num_hidden = 128
num_output = 10
batch_size = 64

test_iter = DataLoader(X_test, y_test, batch_size)
network = mlp(num_input, num_hidden, num_output, lr=0, l2=0)
param = np.load('//Users/wangjing/Desktop/hw1/param.npy', allow_pickle=True)
network.load_state_dict(param)
network.cuda()

accuracy = 0
for X, y in test_iter:
    X = np.asarray(X)
    y = np.asarray(y)
    y_hat = network(X)
    accuracy += (np.argmax(y_hat, axis=1) == y).sum()
print('accuracy:', accuracy / len(test_iter))