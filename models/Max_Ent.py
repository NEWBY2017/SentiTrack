import random
import math

class Max_Entropy():
    def __init__(self, train, num_of_epoch=500, theta_default = None):
        if theta_default == None:
            vocab = set()
            for X, y in train:
                for x in X:
                    vocab.add(x)
            theta = {i: 0 for i in vocab}
        else:
            theta = theta_default

        lr = orig_lr = 0.1
        lamb = 0.01

        pred = self.pred
        tol = 1e-6
        prev_tc = 1e16
        for epoch in range(num_of_epoch):
            for X, y in random.sample(train, len(train)):
                y = 0 if y == "bear" else 1
                # pred
                p = pred(X)
                # c = -y * math.log(p) - (1-y) * math.log(1-p)
                dc_dp = -y/p + (1-y)/(1-p)
                dp_ds = p * (1-p)
                for w in X:
                    theta[w] += dc_dp * dp_ds * lr - lamb * theta[w]
            lr = orig_lr * (1 - epoch /num_of_epoch)

            tc = 0
            for X, y in train:
                y = 0 if y == "bear" else 1
                # pred
                p = pred(X)
                tc += -y * math.log(p) if y == 1 else - (1-y) * math.log(1-p)

            if epoch % 20 == 0:
                print(epoch, tc)
        self.theta = theta

    def pred(self, W):
        s = 0
        for w in W:
            try:
                s += self.theta[w]
            except:
                continue
        p = 1 / (1 + math.exp(s))
        if p==0:p=0.00001
        if p==1:p=0.99999
        return p

    def pred_label(self, W):
        p = self.pred(W)
        if p > 0.5:
            return "bull"
        elif p < 0.5:
            return "bear"
        else:
            return "neutual"