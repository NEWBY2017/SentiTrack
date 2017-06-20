import html.parser as parser
import re

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

    def loads(self, bull_fp, bear_fp):
        bull = []
        cnt = 0
        with open(bull_fp, "r") as file:
            for line in file:
                bull.append(Tweet(line.strip("\n"), "bull"))
                cnt+=1
                if cnt == 50: break

        bear = []
        with open(bear_fp, "r") as file:
            for line in file:
                bear.append(Tweet(line.strip("\n"), "bear"))
                cnt+=1
                if cnt == 100: break
        self.bull = bull
        self.bear = bear
        self.n_bull = len(self.bull)
        self.n_bear = len(self.bear)
        self.data = bull + bear

    def clean(self, lower=True, sub_url=True, sub_stock=True, sub_num=True, rm_tag=True, rm_at=True,
              rm_sig=True, rm_quote=True, abbrev_expand=True):

        if lower:
            for tweet in self.data:
                tweet.text = tweet.text.lower()

        if sub_url:
            re_url = re.compile("https?://\S*")
            for tweet in self.data:
                tweet.apply(lambda s: re_url.sub("url", s))

        if sub_stock:
            re_stock = re.compile("\$\S*")
            for tweet in self.data:
                tweet.apply(lambda s: re_stock.sub("com", s))

        if sub_num:
            re_numb  = re.compile("[\d\.]+\%*")
            for tweet in self.data:
                tweet.apply(lambda s: re_numb.sub("num", s))

        if rm_tag:
            re_tag = re.compile("#\S*")
            for tweet in self.data:
                tweet.apply(lambda s: re_tag.sub("", s))

        if rm_at:
            re_at    = re.compile("@\S*")
            for tweet in self.data:
                tweet.apply(lambda s: re_at.sub("", s))

        if rm_sig:
            re_sig   = re.compile("\s\W*\s|\s\W*$")
            for tweet in self.data:
                tweet.apply(lambda s: re_sig.sub("", s))

        if rm_quote:
            re_qute  = re.compile("[\"()]")
            for tweet in self.data:
                tweet.apply(lambda s: re_qute.sub("", s))

        if abbrev_expand:
            re_s  = re.compile("\'s")
            re_ll = re.compile("\'ll")
            re_d  = re.compile("\'d")
            re_m  = re.compile("\'m")
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
            tweet.words = re_sep.split(tweet.text) if len(tweet.text) > 0 else []

    def get_train(self, n):
        if n > min(self.n_bull, self.n_bear):
            n = min(self.n_bull, self.n_bear)
        X, y = [], []
        for i in range(n):
            tweet = self.bull[i]
            X.append(tweet.words)
            y.append(tweet.label)

        for i in range(n):
            tweet = self.bear[i]
            X.append(tweet.words)
            y.append(tweet.label)
        return X, y

    def get_test(self, n):
        if n > min(self.n_bull, self.n_bear):
            n = min(self.n_bull, self.n_bear)
        X, y = [], []
        for i in range(self.n_bull-n, self.n_bull):
            tweet = self.bull[i]
            X.append(tweet.words)
            y.append(tweet.label)

        for i in range(self.n_bear-n, self.n_bear):
            tweet = self.bear[i]
            X.append(tweet.words)
            y.append(tweet.label)
        return X, y

if __name__ == '__main__':
    bear_fp = "/Users/fredzheng/Documents/stocktwits/sentiment/Bearish"
    bull_fp = "/Users/fredzheng/Documents/stocktwits/sentiment/Bullish"
    data = Data()
    data.loads(bull_fp, bear_fp)
    data.clean()

