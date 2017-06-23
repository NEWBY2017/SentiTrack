from eval.functions import *
from models.Data import *
from models.Max_Ent import *
from models.Naive_Bayes import *
import pickle

if __name__ == '__main__':
    ## read bull and bear
    bear_fp = "/Users/fredzheng/Documents/stocktwits/sentiment/Bearish"
    bull_fp = "/Users/fredzheng/Documents/stocktwits/sentiment/Bullish"
    data_fp = "/Users/fredzheng/Documents/stocktwits/sentiment/data_object.pkl"
    w2v_fp  = "/Users/fredzheng/Documents/stocktwits/sentiment/word2vec"

    load_existing_object = True
    if load_existing_object:
        con = open(data_fp, "rb")
        data = pickle.load(con)
        con.close()
    else:
        data = Data()
        data.loads(bull_fp, bear_fp, max_n=50000)
        data.clean()
        data.cut_train_and_test(balance=True)
        # data.save(data_fp)

    ## prep data
    tweets_train = data.train
    trainX = [tweet.words for tweet in tweets_train]
    trainY = [tweet.label for tweet in tweets_train]
    train = list(zip(trainX, trainY))

    tweets_test = data.test
    testX = [tweet.words for tweet in tweets_test]
    testY = [tweet.label for tweet in tweets_test]
    test = list(zip(testX, testY))

    ## naive bayes
    nb = Naive_Bayes(train)
    pred_nb = [nb.pred_label(W) for W, y in train]
    eval(pred_nb, trainY)  ## training error   15.29%

    ## maxent
    maxent = Max_Entropy(train, num_of_epoch=100)
    pred_me = [maxent.pred_label(W) for W, y in train]
    eval(pred_me, [y for W, y in train])  ## training error   13.95%

    ## test error
    pred_nb = [nb.pred_label(W) for W, y in test]
    eval(pred_nb, [y for W, y in test])  ## test error   25.67%

    pred_me = [maxent.pred_label(W) for W, y in test]
    eval(pred_me, [y for W, y in test])  ## test error   31.42%
