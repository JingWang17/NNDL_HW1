# coding=gbk
import numpy as np
from mlxtend.data import loadlocal_mnist
from dataloader import DataLoader
from net import mlp
from loss import CrossEntropyLoss
from val import validate
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import os
#load data
X_train, y_train = loadlocal_mnist(
            images_path='/Users/wangjing/Desktop/hw1/data/train-images.idx3-ubyte',
            labels_path='/Users/wangjing/Desktop/hw1/data/train-labels.idx1-ubyte')
X_test, y_test = loadlocal_mnist(
            images_path='/Users/wangjing/Desktop/hw1/data/t10k-images.idx3-ubyte',
            labels_path='/Users/wangjing/Desktop/hw1/data/t10k-labels.idx1-ubyte')

def normalize(x):
    m = np.mean(x,axis=0,keepdims=True)
    var = np.var(x,axis=0,keepdims=True)
    return (x-m)/np.sqrt(var+1e-05)
X_train = normalize(X_train)
X_test = normalize(X_test)
#grid_search
num_input = 28*28
num_hidden_list = [512,256,128]
num_output = 10
lr_list = [0.2,0.1]
l2_list = [1e-04,5e-04]
#epochs = 120
batch_size = 64
d = []

for num_hidden in num_hidden_list:
    for lr in lr_list:
        for l2 in l2_list:
            print(' ')
            print('num_hidden:{},lr:{},l2:{}'.format(num_hidden, lr, l2))

            train_iter = DataLoader(X_train, y_train, batch_size)
            test_iter = DataLoader(X_test, y_test, batch_size)
            network = mlp(num_input, num_hidden, num_output, lr, l2)

            ce_train = []
            ce_test = []
            accuracy = []

            for iteration, data in enumerate(train_iter):
                iteration += 1
                X, y = data
                X = np.asarray(X)
                y = np.asarray(y)
                y_hat = network(X)
                l = CrossEntropyLoss(y_hat, y)
                grad = network.backward(X, y)
                network.step(grad)

                ce_train.append(l.item())
                acc, ce = validate(test_iter, network)
                ce_test.append(ce.item())
                accuracy.append(acc.item())
                network.lr_decay(iteration)

                if iteration % 10 == 0:
                    print('iteration:{},loss:{},accuracy:{}'.format(iteration, ce, acc))

            network.cpu()
            np.save('/Users/wangjing/Desktop/hw1/param_' + str(num_hidden) + '_' + str(lr) + '_' + str(
                l2) + '.npy', network.parameters())
            num = list(range(len(ce_train)))
            plt.figure(figsize=(6, 6), dpi=100)
            plt.plot(num, ce_train, label='trainset loss')
            plt.plot(num, ce_test, label='testset loss')
            plt.xlabel('iteration')
            plt.ylabel('CrossEntropyLoss')
            plt.legend()
            plt.savefig(
                '/Users/wangjing/Desktop/hw1/loss_' + str(num_hidden) + '_' + str(lr) + '_' + str(
                    l2) + '.png')

            plt.figure(figsize=(6, 6), dpi=100)
            plt.plot(num, accuracy)
            plt.xlabel('iteration')
            plt.ylabel('testset accuracy')
            plt.savefig(
                '/Users/wangjing/Desktop/hw1/accuracy_' + str(num_hidden) + '_' + str(lr) + '_' + str(
                    l2) + '.png')

            d.append([num_hidden, lr, l2, acc])

df = pd.DataFrame(d, columns=['num_hidden', 'lr', 'l2', 'accuracy'])
df.to_csv('/Users/wangjing/Desktop/hw1/result.csv', index=False)


