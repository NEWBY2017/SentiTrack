from models.Data import *
from models.RNN import *
from collections import Counter
# from gensim.models import Word2Vec
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
        data.loads(bull_fp, bear_fp, max_n=20000)
        data.clean()
        # data.cut_train_and_test(balance=True)
        # data.save(data_fp)

    w2i = data.word2index()
    data.cut_train_and_test(balance=True)

    ## rnn
    batch_size = 128
    num_batch_per_epoch = data.get_num_of_batch(batch_size)
    hidden_size = 100
    num_epoch = 100
    lr = 0.002
    embedding_size = 100

    ## embedding

    rnn = RNN("gru", hidden_size, embedding_size, lr=lr)
    rnn.build_graph(embedding=True, vocab_size=len(w2i), embedding_size=100)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epoch):
        for _ in range(num_batch_per_epoch):
            batch = data.get_train_batch(batch_size)
            batch_X = [np.array(tweet.word_indexes) for tweet in batch]
            batch_y = [int(tweet.label=="bull") for tweet in batch]
            _, _= rnn.train(batch_X, batch_y, sess, embedding=True)

        ## accuracy check, train, valid, test sets
        train_X = [np.array(tweet.word_indexes) for tweet in data.train]
        train_y = [int(tweet.label == "bull") for tweet in data.train]
        train_ent, train_acc = rnn.cal_accuracy(train_X, train_y, sess, embedding=True)

        valid_X = [np.array(tweet.word_indexes) for tweet in data.valid]
        valid_y = [int(tweet.label == "bull") for tweet in data.valid]
        _, valid_acc = rnn.cal_accuracy(valid_X, valid_y, sess, embedding=True)

        test_X = [np.array(tweet.word_indexes) for tweet in data.test]
        test_y = [int(tweet.label == "bull") for tweet in data.test]
        _, test_acc = rnn.cal_accuracy(test_X, test_y, sess, embedding=True)

        print("Current epoch", epoch, "\tCross entropy", train_ent,
              "\tTraining accuracy", train_acc,
              "\tValidate accuracy", valid_acc,
              "\tTest accuracy", test_acc)
