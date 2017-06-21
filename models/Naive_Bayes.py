from collections import defaultdict
import math


class Naive_Bayes():
    def __init__(self, train):
        pos = defaultdict(lambda: 1)  ## Laplace smoothing
        neg = defaultdict(lambda: 1)  ## Laplace smoothing
        for X, y in train:
            col = pos if y == "bull" else neg
            for x in X:
                col[x] += 1
        pos_sum = sum(pos.values())
        neg_sum = sum(neg.values())
        self.pos = pos = {i: math.log(j / pos_sum) for i, j in pos.items()}
        self.neg = neg = {i: math.log(j / neg_sum) for i, j in neg.items()}
        self.pos["##default##"] = math.log(1 / max(pos_sum, neg_sum))
        self.neg["##default##"] = math.log(1 / max(pos_sum, neg_sum))

    def pred(self, W):
        '''
        :param W: a list of words
        :return: probability of being positive
        '''
        p, n = 0, 0
        pos, neg = self.pos, self.neg
        for w in W:
            p += pos.get(w, pos.get("##default##"))
            n += neg.get(w, neg.get("##default##"))
        m = max(p, n)
        p, n = map(lambda x: math.exp(x - m), [p, n])
        return p / (p + n)

    def pred_label(self, W):
        p = self.pred(W)
        if p > 0.5:
            return "bull"
        elif p < 0.5:
            return "bear"
        else:
            return "neutual"
