# SentiTrack

This project is created for two purposes. The first is to compare models commonly used in sentiment analysis. Another more interesting purpose is to tweak the RNN parameters to see how performance changes.

### Model comparision

The table below shows the training and testing accuracy for various models. Data contains 20000 tweets with bull sentiment and 20000 with bear sentiment. Twitters are preprocessed and divided into 60% for training, 20% for validation, and 20% for testing.

|Model|Train|Test|
|---|---|---|
|NB *  |84.71%| 74.33%|
|MaxEnt*|86.05%| 68.58%|
|LSTM -W2V -H=50 -drop=False |83.27%|73.00%|
|GRU -W2V -H=50 -drop=False|88.81%|71.19%|
|GRU -W2V -H=100 -drop=False|94.81%|71.22%|
|RNN -W2V -H=50 -drop=False|88.56% | 64.97%|
|LSTM_Peep -W2V -H=50 -drop=False| 95.31% | 71.32% |
||||
|GRU -E -H=50 -drop=False|84.42%|72.88%|

\* This accuracy is calculated using 50000 bull and 50000 bear.

***Notation***  
-H: number of nodes in hidden layer  
-E: training model and word embedding at the same time  
-W2V use word2vec as embedding  
-drop: use dropout


***Notes:***   
  1. All RNN models are able to fit training set very well (>92%), the training accuracy is for the model that achieves the best validation accuracy, suggesting that RNNs are able to overfit the training set.  
  2. Judging by GRU -H100 and GRU -H50, increasing complexity might not work if the training set contains 24000 tweets only.

### Planned future works
  1. Compare RNN, LSTM, GRU
  2. Test dropout
  3. Gradient clipping
  4. Generate positive/negative sentences
  5. Save Parameters