from models.Data import *
from models.RNN import *
from gensim.models import Word2Vec
import pickle
import tensorflow as tf
np.set_printoptions(suppress=True)

if __name__ == '__main__':
    ## read bull and bear
    bear_fp = "/Users/fredzheng/Documents/stocktwits/sentiment/Bearish"
    bull_fp = "/Users/fredzheng/Documents/stocktwits/sentiment/Bullish"
    data_fp = "/Users/fredzheng/Documents/stocktwits/sentiment/data_object.pkl"
    w2v_fp  = "/Users/fredzheng/Documents/stocktwits/sentiment/word2vec"

    ## load data
    load_existing_object = False
    if load_existing_object:
        con = open(data_fp, "rb")
        data = pickle.load(con)
        con.close()
    else:
        data = Data()
        data.loads(bull_fp, bear_fp, max_n=50000)
        data.clean()
        # data.cut_train_and_test(balance=True)
        # data.save(data_fp)

    ## Word2Vec embedding
    load_existing_word2vec = True
    embedding_size = 100
    if load_existing_word2vec:
        model = Word2Vec.load(w2v_fp)
    else:
        model = Word2Vec([tweet.words for tweet in data.data], size=embedding_size, window=5, min_count=5)
        model.save(w2v_fp)
    wv = model.wv
    data.filter_words(set(model.wv.vocab.keys()))
    data.cut_train_and_test(balance=True)

    ## rnn
    batch_size = 5000
    num_batch_per_epoch = data.get_num_of_batch(batch_size)
    hidden_size = 50
    num_epoch = 100
    lr = 0.002

    rnn = RNN(hidden_size, embedding_size, lr=lr)
    rnn.build_graph()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epoch):
        for _ in range(num_batch_per_epoch):
            batch = data.get_train_batch(batch_size)
            batch_X = [np.array([wv[w] for w in tweet.words]) for tweet in batch]
            batch_y = [int(tweet.label=="bull") for tweet in batch]
            _, _= rnn.train(batch_X, batch_y, sess)
        train_X = [np.array([wv[w] for w in tweet.words]) for tweet in data.train]
        train_y = [int(tweet.label == "bull") for tweet in data.train]
        train_ent, train_acc = rnn.cal_accuracy(train_X, train_y, sess)

        test_X = [np.array([wv[w] for w in tweet.words]) for tweet in data.test]
        test_y = [int(tweet.label == "bull") for tweet in data.test]
        _, test_acc = rnn.cal_accuracy(test_X, test_y, sess)

        print("Current epoch", epoch, "\tCross entropy", train_ent, "\tTraining accuracy", train_acc, "\tTest accuracy", test_acc)
