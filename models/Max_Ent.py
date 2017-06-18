import random
import math

class Max_Entropy():
    def __init__(self, train, default = None):
        vocab = set()
        for X, y in train:
            for x in X:
                vocab.add(x)
        theta = {i:0 for i in vocab} if default == None else default
        lr = 0.1
        lamb = 0.01
        for iter in range(0):
            for X, y in random.sample(train, len(train)):
                y = 0 if y == "bear" else 1
                # pred
                s = 0
                for w in X:
                    s += theta[w]
                p = 1/(1+math.exp(s))
                if p == 0 or p == 1: continue
                c = -y * math.log(p) - (1-y) * math.log(1-p)
                dc_dp = -y/p + (1-y)/(1-p)
                dp_ds = p * (1-p)
                for w in X:
                    theta[w] += dc_dp * dp_ds * lr - lamb * theta[w]
            lr = 0.1 * (1 - iter /1000)

            tc = 0
            for X, y in train:
                y = 0 if y == "bear" else 1
                # pred
                s = 0
                for w in X:
                    s += theta[w]
                p = 1/(1 + math.exp(s))
                if p == 1 and y == 0:
                    tc += 10
                elif p == 0 and y == 1:
                    tc += 10
                else:
                    tc += -y * math.log(p) if y == 1 else - (1-y) * math.log(1-p)
            print(iter, tc)
        self.theta = theta

    def pred(self, W):
        s = 0
        for w in W:
            try:
                s += self.theta[w]
            except:
                continue
        p = 1 / (1 + math.exp(s))
        return p

    def pred_label(self, W):
        p = self.pred(W)
        if p > 0.5:
            return "bull"
        elif p < 0.5:
            return "bear"
        else:
            return "neutual"