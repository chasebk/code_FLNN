# Data link: 
    https://bitbucket.org/nguyenthieu2102/cro_mlnn_jcss/src/bio_inspired/data/
* column: num_of_conn

# Conclude
LSTM - GPU: 5.5h
GRU - GPU: 2h
BiLSTM - CPU : 31h


rows: 30 000
LSTM - CPU: 3h


LSTM, GRU  =>  SMAPE: 3-4%
MLP ==> SMAPE: 8%

 
## Giang's test on CPU
1. FL-GANN
* SMAPE: 5% => 8%
* Time: 4000s => 7000s (1h < x < 2h)
* The differents are
    * % split dataset (train, valid, test) = (33%, 33%, 33%)
    * Epoch: 700, Pop size: 250
    * Normalize the whole data before split it (This is a problem need to think about it)

2. LSTM
* SMAPE: 
* Time:
* 1 Hidden Layer
* Depend on programing languagues and server architecture (even setup code tensorflow to run on 2 CPU, but its still 
use 8 CPU and not all 100% CPU on server)   ==> Effect Time Consuming


    
     
















