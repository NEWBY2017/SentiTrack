from models.Max_Ent import *
from models.Naive_Bayes import *
from eval.functions import *
from models.preprocess import *

## read bull and bear
bear = "/Users/fredzheng/Documents/stocktwits/sentiment/Bearish"
bull = "/Users/fredzheng/Documents/stocktwits/sentiment/Bullish"

bull = read(bull); bear = read(bear)

trainX = bull[:10000] + bear[:10000]
trainY = ["bull"] * 10000 + ["bear"] * 10000
trainX = parse(trainX)

train = [[trainX[i], trainY[i]] for i in range(len(trainX))]
train = [[content, sent] for content, sent in train if len(content) > 0]

## maxent
maxent = Max_Entropy(train)

## naive bayes
nb = Naive_Bayes(train)
eval([nb.pred_label(W) for W, y in train], [y for W, y in train])       ## training error
'''
[[ 8296.  1240.     0.]
 [ 1440.  8557.     0.]
 [    0.     0.     0.]]
'''

eval([maxent.pred_label(W) for W, y in train], [y for W, y in train])       ## training error

## test
testX = bull[10000:20000] + bear[10000:20000]
testY = ["bull"] * 10000 + ["bear"] * 10000
testX = parse(testX)

test = [[testX[i], testY[i]] for i in range(len(testX))]
test = [[content, sent] for content, sent in test if len(content) > 0]
eval([nb.pred_label(W) for W, y in test], [y for W, y in test])       ## test error

'''
[[ 6630.  2801.     0.]
 [ 2763.  6747.     0.]
 [  250.   243.     0.]]
 '''

eval([maxent.pred_label(W) for W, y in test], [y for W, y in test])       ## test error

'''
[[    1.             0.40854701  1170.        ]
 [    2.             0.69058296  1338.        ]
 [    3.             0.72247557  1535.        ]
 [    4.             0.71456693  1524.        ]
 [    5.             0.71568627  1530.        ]
 [    6.             0.7199211   1521.        ]
 [    7.             0.71787709  1432.        ]
 [    8.             0.69202634  1367.        ]
 [    9.             0.69438029  1299.        ]
 [   10.             0.71634981  1315.        ]
 [   11.             0.6701209   1158.        ]
 [   12.             0.71750433  1154.        ]
 [   13.             0.69508526   997.        ]
 [   14.             0.71558442   770.        ]
 [   15.             0.69464286   560.        ]
 [   16.             0.65299685   317.        ]
 [   17.             0.72972973   222.        ]
 [   18.             0.72307692   130.        ]
 [   19.             0.68627451    51.        ]
 [   20.             0.76          25.        ]
 [   21.             0.66666667     6.        ]
 [   22.             1.             5.        ]
 [   23.             1.             3.        ]
 [   24.             0.5            2.        ]
 [   25.             0.66666667     3.        ]]
'''

'''
----------------max entropy-------------------
[[    1.             0.40940171  1170.        ]
 [    2.             0.68759342  1338.        ]
 [    3.             0.70553746  1535.        ]
 [    4.             0.72112861  1524.        ]
 [    5.             0.7         1530.        ]
 [    6.             0.69296515  1521.        ]
 [    7.             0.69692737  1432.        ]
 [    8.             0.68471105  1367.        ]
 [    9.             0.69053118  1299.        ]
 [   10.             0.7095057   1315.        ]
 [   11.             0.68393782  1158.        ]
 [   12.             0.70710572  1154.        ]
 [   13.             0.67101304   997.        ]
 [   14.             0.71038961   770.        ]
 [   15.             0.69464286   560.        ]
 [   16.             0.67507886   317.        ]
 [   17.             0.72072072   222.        ]
 [   18.             0.67692308   130.        ]
 [   19.             0.70588235    51.        ]
 [   20.             0.68          25.        ]
 [   21.             0.83333333     6.        ]
 [   22.             0.8            5.        ]
 [   23.             1.             3.        ]
 [   24.             0.5            2.        ]
 [   25.             0.66666667     3.        ]]
'''
