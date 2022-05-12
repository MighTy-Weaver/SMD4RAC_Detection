# AC-Detection-Validation

This repository is the second phase of
project [Inefficient-AC-detection](https://github.com/MighTy-Weaver/Inefficient-AC-detection).

We aim to conduct more empirical research by leveraging advanced supervised learning techniques.

First phase will focus on Contrastive Learning, as shown
in [this folder](https://github.com/MighTy-Weaver/AC-Detection-Validation/tree/main/ContrastiveLearning).


lstm 0.39549119013212075 25
bilstm 0.40827862627495326 75
transformer 0.3003556128230538 5
lstm-transformer 0.2245773889655719 75
bilstm-transformer 0.20199863296376408 100

TODO: 
1. 改成随机可重复抽样，data = 40000.
2. Classification Experiments 


Positive examples Negative examples 1:1

Demand data 100k => 50k : 50k