| Architecture | LSTM1 | LSTM2 | Dense | Dropout | Cell | Params | Mean LOSO Acc | Std | Avg Epochs |
|---|---|---|---|---|---|---|---|---|---|
| gru | 64 | 32 | 32 | 0.3 | GRU | 145,159 | 0.970 | 0.020 | 37 |
| high_dropout | 64 | 32 | 32 | 0.5 | LSTM | 192,007 | 0.961 | 0.025 | 44 |
| current (current) | 64 | 32 | 32 | 0.3 | LSTM | 192,007 | 0.960 | 0.025 | 47 |
| smaller | 32 | 16 | 16 | 0.3 | LSTM | 77,063 | 0.958 | 0.030 | 45 |
| single_layer | 64 | - | 32 | 0.3 | LSTM | 152,839 | 0.945 | 0.047 | 43 |
| no_dense_head | 64 | 32 | - | 0.3 | LSTM | 190,151 | 0.939 | 0.057 | 42 |
| bigger | 128 | 64 | 64 | 0.3 | LSTM | 535,559 | 0.921 | 0.067 | 40 |
