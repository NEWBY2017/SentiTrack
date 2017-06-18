import numpy as np
from collections import defaultdict

def eval(pred, label):
    mat = np.zeros([3,3])
    ind = ["bull", "bear", "neutual"]
    for i in range(len(pred)):
        mat[ind.index(pred[i]), ind.index(label[i])] += 1
    print(mat)

def eval2(pred, label, corpus):
    from matplotlib import pyplot as plt
    dmat = defaultdict(lambda : np.zeros([3,3]))
    ind = ["bull", "bear", "neutual"]
    for i in range(len(pred)):
        dmat[len(corpus[i])][ind.index(pred[i]), ind.index(label[i])] += 1

    L = []
    for i, data in dmat.items():
        L.append([i, (data[0,0] + data[1,1])/data.sum(), data.sum()])
    L = np.array(sorted(L, key=lambda x:x[0]))
    print(np.array(sorted(L, key=lambda x:x[0])))
    plt.plot(L[:,0], L[:,1], "o")
    plt.plot(L[:,0], L[:,2]/L[:,2].max())

    plt.show()
