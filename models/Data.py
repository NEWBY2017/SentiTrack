import html.parser as parser
import re
import numpy as np
import random


class Tweet():
    def __init__(self, text, label):
        text = parser.unescape(text)
        self.raw = text
        self.text = text
        self.label = label

    def apply(self, func):
        self.text = func(self.text)

    def __repr__(self):
        return "Content: " + self.raw + "\tLabel: " + self.label

    def print(self):
        print(self.text)


class Data():
    '''
    Store, preprocess, and output the data with sentiment
    '''

    def __init__(self):
        pass

    def loads(self, bull_fp, bear_fp, max_n=None):
        bull = []
        if max_n == None: max_n = 1e20
        cnt = 0
        with open(bull_fp, "r") as file:
            for line in file:
                bull.append(Tweet(line.strip("\n"), "bull"))
                cnt += 1
                if cnt == max_n: break

        bear = []
        cnt = 0
        with open(bear_fp, "r") as file:
            for line in file:
                bear.append(Tweet(line.strip("\n"), "bear"))
                cnt += 1
                if cnt == max_n: break
        self.bull = bull
        self.bear = bear
        self.n_bull = len(self.bull)
        self.n_bear = len(self.bear)
        self.data = bull + bear

    def save(self, fp):
        import pickle
        con = open(fp, "wb")
        pickle.dump(self, con)
        con.close()

    def clean(self, lower=True, sub_url=True, sub_stock=True, sub_num=True, rm_tag=True, rm_at=True,
              rm_sig=True, rm_quote=True, abbrev_expand=True, remove_empty=True, remove_stopwords=False,
              stoppath=None):

        if lower:
            for tweet in self.data:
                tweet.text = tweet.text.lower()

        if sub_url:
            re_url = re.compile("https?://\S*")
            for tweet in self.data:
                tweet.apply(lambda s: re_url.sub(" url ", s))

        if sub_stock:
            re_stock = re.compile("\$\S*")
            for tweet in self.data:
                tweet.apply(lambda s: re_stock.sub(" com ", s))

        if sub_num:
            re_numb = re.compile("\d+\.\d*%*|\d*\.\d+\%*|[\d]+%*")
            for tweet in self.data:
                tweet.apply(lambda s: re_numb.sub(" num ", s))

        if rm_tag:
            re_tag = re.compile("#\S*")
            for tweet in self.data:
                tweet.apply(lambda s: re_tag.sub("", s))

        if rm_at:
            re_at = re.compile("@\S*")
            for tweet in self.data:
                tweet.apply(lambda s: re_at.sub("", s))

        re_space = re.compile("\s+")
        for tweet in self.data:
            tweet.apply(lambda s: re_space.sub(" ", s))

        if rm_sig:
            re_sig = re.compile("\s\W+\s|\s\W+$")
            for tweet in self.data:
                tweet.apply(lambda s: re_sig.sub("", s))

        if rm_quote:
            re_qute = re.compile("[\"()]")
            for tweet in self.data:
                tweet.apply(lambda s: re_qute.sub("", s))

        if abbrev_expand:
            re_s = re.compile("\'s")
            re_ll = re.compile("\'ll")
            re_d = re.compile("\'d")
            re_m = re.compile("\'m")
            re_ve = re.compile("\'ve")
            re_re = re.compile("\'re")
            re_nt = re.compile("n\'t")
            for tweet in self.data:
                tweet.apply(lambda s: re_s.sub(" is ", s))
                tweet.apply(lambda s: re_d.sub(" would ", s))
                tweet.apply(lambda s: re_m.sub(" am ", s))
                tweet.apply(lambda s: re_ll.sub(" will ", s))
                tweet.apply(lambda s: re_ve.sub(" have ", s))
                tweet.apply(lambda s: re_re.sub(" are ", s))
                tweet.apply(lambda s: re_nt.sub(" not ", s))

        ## TODO: consider punctuation
        re_sep = re.compile("\W+")
        for tweet in self.data:
            tweet.words = [word for word in re_sep.split(tweet.text) if word!=""] if len(tweet.text) > 0 else []

        if remove_stopwords:
            with open(stoppath, "r") as file:
                stopwords = {line.strip("\n") for line in file}
            for tweet in self.data:
                tweet.words = [word for word in tweet.words if word not in stopwords]

        if remove_empty:
            self.bull = [tweet for tweet in self.bull if len(set(tweet.words).difference("url", "num", "com")) > 2]
            self.bear = [tweet for tweet in self.bear if len(set(tweet.words).difference("url", "num", "com")) > 2]
            self.n_bull, self.n_bear = len(self.bull), len(self.bear)
            self.data = self.bull + self.bear

    def filter_words(self, word_set):
        for tweet in self.data:
            tweet.words = [word for word in tweet.words if word in word_set]
        self.bull = [tweet for tweet in self.bull if len(set(tweet.words).difference("url", "num", "com")) > 2]
        self.bear = [tweet for tweet in self.bear if len(set(tweet.words).difference("url", "num", "com")) > 2]
        self.n_bull, self.n_bear = len(self.bull), len(self.bear)
        self.data = self.bull + self.bear

    def cut_train_and_test(self, balance=True):
        if balance:
            n_bull, n_bear = [min(self.n_bull, self.n_bear)] * 2
        else:
            n_bull, n_bear = self.n_bull, self.n_bear
        bull, bear = np.array(self.bull)[:n_bull], np.array(self.bear)[:n_bear]

        np.random.seed(123)
        bull_ind = np.random.choice(["train", "valid", "test"], n_bull, p=[0.6, 0.2, 0.2])
        bear_ind = np.random.choice(["train", "valid", "test"], n_bear, p=[0.6, 0.2, 0.2])

        self.train = np.r_[bull[bull_ind == "train"], bear[bear_ind == "train"]].tolist()
        self.valid = np.r_[bull[bull_ind == "valid"], bear[bear_ind == "valid"]].tolist()
        self.test = np.r_[bull[bull_ind == "test"], bear[bear_ind == "test"]].tolist()

    def get_num_of_batch(self, batch_size=500):
        return len(self.train) // batch_size

    def get_train_batch(self, batch_size=500):
        if "random_train" not in dir(self):
            self.random_train = random.sample(self.train, len(self.train))
            self.pointer = 0

        batch = self.random_train[self.pointer:self.pointer + batch_size]
        self.pointer += batch_size

        if self.pointer + batch_size >= len(self.train):
            del self.random_train
        return batch

    def words_dict(self):
        from collections import defaultdict

        counter = defaultdict(int)
        for tweet in self.data:
            for word in set(tweet.words):
                counter[word] += 1
        return counter

    def word2index(self):
        d = self.words_dict()
        d = sorted([(i, j) for i, j in d.items() if j >= 5], key=lambda x: x[1], reverse=True)
        w2i = {j[0]: i for i, j in enumerate(d)}
        self.filter_words(set(w2i.keys()))
        for tweet in self.data:
            tweet.word_indexes = [w2i[w] for w in tweet.words]
        return w2i


if __name__ == '__main__':
    bear_fp = "/Users/fredzheng/Documents/stocktwits/sentiment/Bearish"
    bull_fp = "/Users/fredzheng/Documents/stocktwits/sentiment/Bullish"
    data = Data()
    data.loads(bull_fp, bear_fp, max_n=1000)
    data.clean()
    data.cut_train_and_test(balance=True)
