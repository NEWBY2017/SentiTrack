import numpy as np
import tensorflow as tf

def cut_and_pad(X, max_length):
    '''
    Pad the sequence if the sequence length is less than max_length, else cut the sequence
    :param X: a list of numpy array, each array is [max_length, embedding_size]
    :return: a numpy array shaped as [batch_size, max_length, embedding_size]
    '''
    new_X = []
    for x in X:
        n_row, n_col = x.shape
        if n_row >= max_length:
            x = x[:max_length]
        else:
            x = np.r_[x, np.zeros([ max_length -n_row, n_col])]
        new_X.append(x)
    return np.array(new_X)

class RNN():
    def __init__(self, type="lstm", hidden_size=50, embedding_size=50, lr=0.01):
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lr = lr
        self.type=type

    def build_graph(self, keep_prob=1, gradient_clipping=False):
        '''
        The parameters are not in use yet.
        :param type: {"lstm", "gru", "vanila"}
        :param dropout: apply dropout
        :param gradient_clipping: apply gradient clipping
        '''
        self.X       = tf.placeholder(tf.float32, [None, None, self.embedding_size])
        self.Y       = tf.placeholder(tf.float32, [None, 2])
        self.seq_len = tf.placeholder(tf.float32, [None])

        with tf.variable_scope("RNN"):
            ## Rnn
            if self.type == "lstm":
                cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            elif self.type == "gru":
                cell =  tf.contrib.rnn.GRUCell(self.hidden_size)
            elif self.type == "rnn":
                cell = tf.contrib.rnn.BasicRNNCell(self.hidden_size)
            elif self.type == "lstm_peep":
                cell = tf.contrib.rnn.LSTMCell(self.hidden_size, use_peepholes=True)
            else:
                raise ValueError("Cell type cannot be %s" %type)

            if keep_prob<1:
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
            ## outputs shape: [batch_size, max_length, hidden_size]
            ## last_states shape: c=[batch_size, hidden_size]
            outputs, last_states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32, sequence_length=self.seq_len)

        with tf.variable_scope("FC"):
            ## prediction - a FC layer
            ## bull=1, bear=0
            self.W = tf.Variable(tf.truncated_normal([self.hidden_size, 2]))
            self.b = tf.Variable(tf.truncated_normal([2]))
            if self.type in ["lstm", "lstm_peep"]:
                last_states = last_states.c
            self.pred = tf.nn.softmax(tf.matmul(last_states, self.W) + self.b)

        with tf.variable_scope("Cross_entropy_optimization"):
            self.cross_entropy = -tf.reduce_sum(self.Y * tf.log(self.pred))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.Y,1)), "float"))
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

    def preprocess_X(self, X, max_length_allowed=50):
        seq_length = [min(len(embeddings), max_length_allowed) for embeddings in X]
        max_length = max(seq_length)
        X = cut_and_pad(X, max_length).astype(np.float32)
        return X, seq_length

    def preproecss_y(self, y):
        Y = np.zeros([len(y), 2])
        for i, j in enumerate(y): Y[i, j] = 1
        return Y

    def train(self, batch_X, batch_y, sess):
        batch_X, seq_len = self.preprocess_X(batch_X)
        batch_Y = self.preproecss_y(batch_y)
        entropy, _, train_acc = sess.run([self.cross_entropy, self.optimizer, self.accuracy],
                     feed_dict={self.X:batch_X, self.Y:batch_Y, self.seq_len:seq_len})
        return entropy, train_acc

    def predict(self, X, sess):
        X, seq_len = self.preprocess_X(X)
        pred = sess.run(self.pred, feed_dict={self.X:X, self.seq_len:seq_len})
        return pred

    def cal_accuracy(self, X, y, sess):
        X, seq_len = self.preprocess_X(X)
        Y = self.preproecss_y(y)
        entropy, train_acc = sess.run([self.cross_entropy, self.accuracy],
                     feed_dict={self.X:X, self.Y:Y, self.seq_len:seq_len})
        return entropy, train_acc


