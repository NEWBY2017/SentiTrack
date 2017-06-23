# SentiTrack

This project is created for two purposes. The first is to compare models commonly used in sentiment analysis. Another more interesting purpose is to tweak the RNN parameters to see how performance changes.

### Model comparision

The table below shows the training and testing accuracy for various models. 

|Model|Train|Test|
|---|---|---|
|NB   |84.71%| 74.33%|
|MaxEnt|86.05%| 68.58%|
|LSTM -H=50 -drop=False *|71.93%|69.26%|

\* This accuracy is obtained at epoch 20. The model may not converge because both training and testing accuracy are still increasing. 
