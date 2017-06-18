import html.parser as parser
import nltk

## read stopwords
stop_words_fp = "/Users/fredzheng/Documents/stocktwits/stopwords"
stop_words = set()
with open(stop_words_fp) as file:
    for line in file:
        stop_words.add(line.strip("\n"))

def read(fp):
    data = []
    with open(fp, "r") as file:
        for line in file:
            data.append(parser.unescape(line.strip("\n")))
    return data

def is_decimal(s):
    try:
        float(s)
        return True
    except:
        return False

def is_number(s):
    out = ""
    if s.startswith("~"):
        s = s[1:]

    if s.startswith("-"):
        out += "#neg_"
        s = s[1:]
    elif s.startswith("+"):
        out += "#pos_"
        s = s[1:]
    else:
        out += "#pos_"

    if "/" in s:
        i = s.find("/")
        p = s[:i]
        n = s[(i+1):]
        if is_decimal(p) and is_decimal(n):
            suffix = "fraction"
        else:
            return s
    elif is_decimal(s):
        suffix = "number"
    elif is_decimal(s[:-1]) and s[-1] == "%":
        suffix = "percent"
    else:
        return s
    return out + suffix

def short(l):
    new_l = []
    for s in l:
        if s[-2:] == "'d":
            new_l.append(s[:-2])
            new_l.append("would")
        elif s[-3:] == "'ll":
            new_l.append(s[:-3])
            new_l.append("will")
        elif s[-2:] == "'s":
            new_l.append(s[:-2])
            new_l.append("is")
        elif s == "I'm" or s == "i'm":
            new_l.append("i")
            new_l.append("am")
        elif s[-3:] == "'ve":
            new_l.append(s[:-3])
            new_l.append("have")
        elif s[-3:] == "'re":
            new_l.append(s[:-3])
            new_l.append("are")
        elif s[-3:] == "n't":
            new_l.append(s[:-3])
            new_l.append("not")
        else:
            new_l.append(s)
    return  new_l

def parse(corpus):
    trainX = [line.replace(",", " ").replace("\"", " ").replace(")", " ").replace("(", " ").replace("!", " ").replace("?", " ").replace("&", " ").replace("..", " ") for line in corpus]
    trainX = [line.split(" ") for line in trainX]
    trainX = [[entry for entry in line if (not entry.startswith("$") and not entry.startswith("@") and not entry.startswith("http") and entry!="")] for line in trainX]
    trainX = [[entry.strip(".").strip("'").strip("\"") for entry in line] for line in trainX]
    trainX = [[is_number(entry) for entry in line] for line in trainX]
    trainX = [[entry.lower() for entry in line] for line in trainX]
    trainX = [short(line) for line in trainX]
    trainX = [[entry for entry in line if len(entry) > 1 and entry not in stop_words] for line in trainX]
    return trainX
