# SentiTrack

This project is created for two purposes. The first is to compare models commonly used in sentiment analysis. Another more interesting purpose is to tweak the RNN parameters to see how performance changes.

### Model comparision

The table below shows the training and testing accuracy for various models. 

|Model|Train|Test|
|---|---|---|
|NB   |84.71%| 74.33%|
|MaxEnt|86.05%| 68.58%|
|LSTM -H=50 -drop=False *|83.27%|73.00%|

\* This accuracy is obtained at epoch 43. The validation accuracy reaches a plateau. After epoch 61, the model starts to overfit.

### Planned future works
  1. Compare RNN, LSTM, GRU
  2. Test dropout
  3. Gradient clipping
  4. Generate positive/negative sentences
  5. Save Parameters