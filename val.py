import numpy as np
from loss import CrossEntropyLoss
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