import numpy as np

def eval(pred, label):
    mat = np.zeros([3,3])
    ind = ["bull", "bear", "neutual"]
    for i in range(len(pred)):
        mat[ind.index(pred[i]), ind.index(label[i])] += 1
    print("Confusion Matrix: [bull, bear, neutual], (pred, label)")
    print(mat)
    print("\nAccuracy")
    print((np.array(mat)*np.eye(3)).sum()/np.array(mat).sum())

def eval2(pred, label, length):
    import seaborn as sns
    import pandas as pd

    df = pd.DataFrame({"pred": pred, "label": label, "length": length})
    df["equal"] = df["pred"] == df["label"]

    L = []
    for size, sub in df.groupby(df["length"]):
        L.append([size, sub["equal"].sum()/len(sub), len(sub)])
    L = np.array(sorted(L, key=lambda x:x[0]))
    L = L[L[:,2]> 50]
    sns.plt.plot(L[:,0], L[:,1])
    sns.plt.ylim([0,1])
    sns.plt.show()