#!/bin/bash
#1.Here is a sample train script from training model using data of t1 without word2vec
python train_baseline.py --tgt_market t1 --src_markets none --tgt_market_valid DATA/t1/valid_run.tsv --tgt_market_test DATA/t1/test_run.tsv --exp_name lightgcn_without --num_epoch 60 --cuda --model_name LightGCN_no_w2v

#2.Here is a sample train script from training model using data of t2 without word2vec
python train_baseline.py --tgt_market t2 --src_markets none --tgt_market_valid DATA/t2/valid_run.tsv --tgt_market_test DATA/t2/test_run.tsv --exp_name lightgcn_without --num_epoch 50 --cuda --tgt_market_valid_pos DATA/t2/valid_qrel.tsv --model_name LightGCN_no_w2v

#3.Here is a sample train script from training model using data of t1 with word2vec
python train_baseline.py --tgt_market t1 --src_markets none --tgt_market_valid DATA/t1/valid_run.tsv --tgt_market_test DATA/t1/test_run.tsv --exp_name lightgcn_with --num_epoch 60 --cuda --model_name LightGCN_with_w2v

#4.Here is a sample train script from training model using data of t2 with word2vec
python train_baseline.py --tgt_market t2 --src_markets none --tgt_market_valid DATA/t2/valid_run.tsv --tgt_market_test DATA/t2/test_run.tsv --exp_name lightgcn_with --num_epoch 50 --cuda --tgt_market_valid_pos DATA/t2/valid_qrel.tsv --model_name LightGCN_with_w2v

#5.recall feature
python recall_feature.py
python recall_feature_2.py

#6.lightgbm Classifier model   # file: ./merge/result_lgb/t1/
python class_lgb_model.py

#7.xgboost Classifier model    # file: ./merge/result_xgb/t1/
python class_xgb_model.py

#8.catboost Classifier model   # file: ./merge/result_cat/t1/
python class_cat_model.py

#9.lightgbm Rank model         # file: ./merge/result_rank1/t1/    ./merge/result_rank2/t1/
python rank_lgb_model.py

#10.merge result
python trans_file.py
