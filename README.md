# WSDM 2022 CUP - Cross-Market Recommendation - LightGCN modified by Starter Kit 
We totally followed the structure of xmrec's sample code for out own model design. Thus it is very easy to reproduce for version of lightgcn.

## Requirements:
The requirements of experiment environment are listed in requirements.txt.

## Train LightGCN model:

Before running train script, you should perform preprocessing by running script `merge_train.py`. After that, train script is easy to run by referring to the samples below.

`train_baseline.py` is the script for training our model that is taking one target market. 

Here is a sample train script from training model using data of t1 without word2vec:

    python train_baseline.py --tgt_market t1 --src_markets none --tgt_market_valid DATA/t1/valid_run.tsv --tgt_market_test DATA/t1/test_run.tsv --exp_name lightgcn_without --num_epoch 60 --cuda --model_name LightGCN_no_w2v

Here is a sample train script from training model using data of t2 without word2vec:

```
python train_baseline.py --tgt_market t2 --src_markets none --tgt_market_valid DATA/t2/valid_run.tsv --tgt_market_test DATA/t2/test_run.tsv --exp_name lightgcn_without --num_epoch 50 --cuda --tgt_market_valid_pos DATA/t2/valid_qrel.tsv --model_name LightGCN_no_w2v
```

Here is a sample train script from training model using data of t1 with word2vec:

```
python train_baseline.py --tgt_market t1 --src_markets none --tgt_market_valid DATA/t1/valid_run.tsv --tgt_market_test DATA/t1/test_run.tsv --exp_name lightgcn_with --num_epoch 60 --cuda --model_name LightGCN_with_w2v
```

Here is a sample train script from training model using data of t2 with word2vec:

```
python train_baseline.py --tgt_market t2 --src_markets none --tgt_market_valid DATA/t2/valid_run.tsv --tgt_market_test DATA/t2/test_run.tsv --exp_name lightgcn_with --num_epoch 50 --cuda --tgt_market_valid_pos DATA/t2/valid_qrel.tsv --model_name LightGCN_with_w2v
```

After training your model, the scripts prints the directories of model and index checkpoints as well as the run files for the validation and test data as below. 

    Run output files:
    --validation: ./merge/t1/LightGCN_with_w2v_valid.tsv
    --test: ./merge/t1/LightGCN_with_w2v_test.tsv
    Experiment finished successfully!

After getting corresponding result files, result files will be input into other tree-like model as ranking features.

