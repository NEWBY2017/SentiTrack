import json

outs = {}
with open("/Users/fredzheng/Documents/stocktwits/sentiment/all_data_with_sentiment") as file:
    for orig in file:
        line = json.loads(orig)
        sent = line['entities']['sentiment']["basic"]
        if sent not in outs: outs[sent] = open("/Users/fredzheng/Documents/stocktwits/sentiment/%s" % sent, "w")
        out = outs[sent]
        out.write(line["body"])
        out.write("\n")
for o in outs.values():
    o.close()
