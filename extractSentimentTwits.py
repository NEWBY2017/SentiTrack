import gzip
import json
import os

directory = "/Users/fredzheng/Documents/stocktwits/data/"
outDir = "/Users/fredzheng/Documents/stocktwits/sentiment/all_data_with_sentiment"

out = open(outDir, "wb")
filenames = os.listdir(directory)
for filename in filenames:
    if filename.startswith("."): continue
    print(filename)
    with gzip.open(directory + filename, "r") as file:
        for raw in file:
            line = json.loads(raw)
            if line['entities']['sentiment'] == None: continue
            out.write(raw)
out.close()