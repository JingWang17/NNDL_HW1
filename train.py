# coding=gbk
import numpy as np
from mlxtend.data import loadlocal_mnist
from loss import CrossEntropyLoss
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import os
#导入数据集
X_train, y_train = loadlocal_mnist(
            images_path='/Users/wangjing/Desktop/hw1/data/train-images.idx3-ubyte',
            labels_path='/Users/wangjing/Desktop/hw1/data/train-labels.idx1-ubyte')
X_test, y_test = loadlocal_mnist(
            images_path='/Users/wangjing/Desktop/hw1/data/t10k-images.idx3-ubyte',
            labels_path='/Users/wangjing/Desktop/hw1/data/t10k-labels.idx1-ubyte')
#标准化
def normalize(x):
    m = np.mean(x,axis=0,keepdims=True)
    var = np.var(x,axis=0,keepdims=True)
    return (x-m)/np.sqrt(var+1e-05)
X_train = normalize(X_train)
X_test = normalize(X_test)
#网络参数
num_input = 28*28
num_hidden_list = [128,64,32]
num_output = 10
lr_list = [0.2,0.1]
l2_list = [1e-04,5e-04]
batch_size = 64
d = []
#数据处理
class DataLoader(object):
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.length = len(y)
        self.arr = np.array(range(self.length))
        self.batch_size = batch_size

    def __iter__(self):
        self.num = 0
        self.seq = np.random.permutation(self.arr)
        return self

    def __next__(self):
        if self.num + self.batch_size <= self.length:
            sample = self.seq[self.num:(self.num + self.batch_size)]
            self.image = self.X[sample]
            self.label = self.y[sample]
            self.num += self.batch_size
            return self.image, self.label
        else:
            raise StopIteration

    def __len__(self):
        return len(self.y)
#网络搭建
# 线性层
class linear(object):
    def __init__(self, num_input, num_output):
        self.weight = np.random.normal(loc=0, scale=0.01, size=(num_input, num_output))
        self.bias = np.zeros((1, num_output))

    def forward(self, X):
        Y = X @ self.weight + self.bias
        return Y

    def __call__(self, X):
        return self.forward(X)

    def parameters(self):
        return [self.weight, self.bias]

    def load_state_dict(self, param):
        self.weight = param[0]
        self.bias = param[1]



    def cuda(self):
        self.weight = np.asarray(self.weight)
        self.bias = np.asarray(self.bias)


# 激活函数relu
class relu(object):
    def __init__(self):
        pass

    def forward(self, X):
        return (np.abs(X) + X) / 2

    def __call__(self, X):
        return self.forward(X)

    # 两层mlp


class mlp(object):
    def __init__(self, num_input, num_hidden, num_output, lr, l2, milestone=500, gamma=0.5):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.fc1 = linear(num_input, num_hidden)
        self.relu = relu()
        self.fc2 = linear(num_hidden, num_output)
        self.H = 0
        self.Z = 0
        self.K = 0

        self.lr = lr
        # 学习率下降
        self.milestone = milestone
        self.gamma = gamma
        self.lr_ = lr
        self.l2 = l2

    def forward(self, X):
        Y = self.fc1(X)
        self.H = Y
        Y = self.relu(Y)
        self.Z = Y
        Y = self.fc2(Y)
        self.K = np.exp(Y)
        return Y

    def __call__(self, X):
        return self.forward(X)

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

    def load_state_dict(self, param):
        self.fc1.load_state_dict(param[:2])
        self.fc2.load_state_dict(param[2:])

    def backward(self, X, y):
        grad = [0] * 4
        for i in range(len(X)):
            x = X[i]
            z = self.Z[i]
            k = self.K[i]
            k_diag = np.diag(k)
            h = self.H[i]
            h_diag = np.diag(np.where(h > 0, 1., 0.))
            y_hot = np.eye(10)[y[i]]
            e = np.ones((10, 1))
            df = 1 / (k @ e) * e.T - 1 / (k @ y_hot.T) * y_hot
            dk = k_diag @ self.fc2.weight.T
            # w1的梯度
            g = []
            for j in range(len(x)):
                g.append(x[j] * h_diag)
            g = np.hstack(g)
            grad[0] += (df @ dk @ g).T.reshape(self.num_input, self.num_hidden)
            # b1的梯度
            grad[1] += df @ dk @ h_diag
            # w2的梯度
            g = []
            for j in range(len(z)):
                g.append(z[j] * k_diag)
            g = np.hstack(g)
            grad[2] += (g.T @ df.T).reshape(self.num_hidden, self.num_output)
            # b2的梯度
            grad[3] += df @ k_diag
        for i in range(4):
            grad[i] = grad[i] / len(X)

        grad[0] = grad[0] + 2 * self.l2 * self.fc1.weight
        grad[2] = grad[2] + 2 * self.l2 * self.fc2.weight
        return grad

    # 梯度下降
    def step(self, grad):
        self.fc1.weight -= self.lr_ * grad[0]
        self.fc1.bias -= self.lr_ * grad[1]
        self.fc2.weight -= self.lr_ * grad[2]
        self.fc2.bias -= self.lr_ * grad[3]

    # 学习率衰减
    def lr_decay(self, epoch):
        n = int(epoch / self.milestone)
        self.lr_ = self.lr * (self.gamma ** n)



    def cuda(self):
        self.fc1.cuda()
        self.fc2.cuda()

def validate(iter_,net):
    accuracy = 0
    l = []
    for X,y in iter_:
        X = np.asarray(X)
        y = np.asarray(y)
        y_hat = net(X)
        loss = CrossEntropyLoss(y_hat,y)
        accuracy += (np.argmax(y_hat,axis=1)==y).sum()
        l.append(loss.item())
    return accuracy/len(iter_), np.mean(l)

for num_hidden in num_hidden_list:
    for lr in lr_list:
        for l2 in l2_list:
            print(' ')
            print('num:{},lr:{},l2:{}'.format(num_hidden, lr, l2))

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


            np.save('/Users/wangjing/Desktop/hw1/param_' + str(num_hidden) + '_' + str(lr) + '_' + str(
                l2) + '.npy', network.parameters())
            num = list(range(len(ce_train)))
            plt.figure(figsize=(6, 6), dpi=100)
            plt.plot(num, ce_train, label='trainset loss')
            plt.plot(num, ce_test, label='testset loss')
            plt.xlabel('iteration')
            plt.ylabel('CrossEntropyLoss')
            plt.legend()
            plt.savefig('/Users/wangjing/Desktop/hw1/loss_' + str(num_hidden) + '_' + str(lr) + '_' + str(
                    l2) + '.png')

            plt.figure(figsize=(6, 6), dpi=100)
            plt.plot(num, accuracy)
            plt.xlabel('iteration')
            plt.ylabel('testset accuracy')
            plt.savefig(
                '/Users/wangjing/Desktop/hw1/accuracy_' + str(num_hidden) + '_' + str(lr) + '_' + str(
                    l2) + '.png')

            d.append([num, lr, l2, acc])

df = pd.DataFrame(d, columns=['num_hidden', 'lr', 'l2', 'accuracy'])
df.to_csv('/Users/wangjing/Desktop/hw1/result.csv', index=False)


