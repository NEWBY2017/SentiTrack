from eval.functions import *
from models.Data import *
from gensim.models import Word2Vec
import pickle

if __name__ == '__main__':
    ## read bull and bear
    bear_fp = "/Users/fredzheng/Documents/stocktwits/sentiment/Bearish"
    bull_fp = "/Users/fredzheng/Documents/stocktwits/sentiment/Bullish"
    data_fp = "/Users/fredzheng/Documents/stocktwits/sentiment/data_object.pkl"
    w2v_fp  = "/Users/fredzheng/Documents/stocktwits/sentiment/word2vec"

    load_existing_object = False
    if load_existing_object:
        con = open(data_fp, "rb")
        data = pickle.load(con)
        con.close()
    else:
        data = Data()
        data.loads(bull_fp, bear_fp, max_n=5000)
        data.clean()
        data.cut_train_and_test(balance=True)
        # data.save(data_fp)

    load_existing_word2vec = True
    if load_existing_word2vec:
        model = Word2Vec.load(w2v_fp)
    else:
        model = Word2Vec([tweet.words for tweet in data.data], size=100, window=5, min_count=5)
        model.save(w2v_fp)



    ## prep data
    tweets_train = data.train
    trainX = [tweet.words for tweet in tweets_train]
    trainY = [tweet.label for tweet in tweets_train]
    train = list(zip(trainX, trainY))

    tweets_test = data.test
    testX = [tweet.words for tweet in tweets_test]
    testY = [tweet.label for tweet in tweets_test]
    test = list(zip(testX, testY))
