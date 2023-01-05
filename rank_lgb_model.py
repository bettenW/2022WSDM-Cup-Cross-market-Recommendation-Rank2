import pandas as pd
import os
import gc
import math
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
import copy

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

from tools import *

filename = './DATA/s1/'
train_s1 = pd.read_csv(filename+'train.tsv', sep='\t')
train_5core_s1 = pd.read_csv(filename+'train_5core.tsv', sep='\t')
valid_qrel_s1 = pd.read_csv(filename+'valid_qrel.tsv', sep='\t') # 验证集 正样本
valid_run_s1 = pd.read_csv(filename+'valid_run.tsv', sep='\t', header=None) # 验证样本
valid_run_s1.columns = ['userId','itemIds']
train_s1['source'] = 0
train_5core_s1['rating']*=5
train_5core_s1['source'] = 1
train_s1 = pd.concat([train_s1,train_5core_s1])

filename = './DATA/s2/'
train_s2 = pd.read_csv(filename+'train.tsv', sep='\t')
train_5core_s2 = pd.read_csv(filename+'train_5core.tsv', sep='\t')
valid_qrel_s2 = pd.read_csv(filename+'valid_qrel.tsv', sep='\t') # 验证集 正样本
valid_run_s2 = pd.read_csv(filename+'valid_run.tsv', sep='\t', header=None) # 验证样本
valid_run_s2.columns = ['userId','itemIds']
train_s2['source'] = 0
train_5core_s2['rating']*=5
train_5core_s2['source'] = 1
train_s2 = pd.concat([train_s2,train_5core_s2])

filename = './DATA/s3/'
train_s3 = pd.read_csv(filename+'train.tsv', sep='\t')
train_5core_s3 = pd.read_csv(filename+'train_5core.tsv', sep='\t')
valid_qrel_s3 = pd.read_csv(filename+'valid_qrel.tsv', sep='\t') # 验证集 正样本
valid_run_s3 = pd.read_csv(filename+'valid_run.tsv', sep='\t', header=None) # 验证样本
valid_run_s3.columns = ['userId','itemIds']
train_s3['source'] = 0
train_5core_s3['rating']*=5
train_5core_s3['source'] = 1
train_s3 = pd.concat([train_s3,train_5core_s3])

filename = './DATA/t1/'
train_t1 = pd.read_csv(filename+'train.tsv', sep='\t')
train_5core_t1 = pd.read_csv(filename+'train_5core.tsv', sep='\t')
valid_qrel_t1 = pd.read_csv(filename+'valid_qrel.tsv', sep='\t') # 验证集 正样本
valid_run_t1 = pd.read_csv(filename+'valid_run.tsv', sep='\t', header=None) # 验证样本
valid_run_t1.columns = ['userId','itemIds']
test_run_t1 = pd.read_csv(filename+'test_run.tsv', sep='\t', header=None) # 测试样本
test_run_t1.columns = ['userId','itemIds']
train_t1['source'] = 0
train_5core_t1['rating']*=5
train_5core_t1['source'] = 1
train1 = pd.concat([train_t1,train_5core_t1])

filename = './DATA/t2/'
train_t2 = pd.read_csv(filename+'train.tsv', sep='\t')
train_5core_t2 = pd.read_csv(filename+'train_5core.tsv', sep='\t')
valid_qrel_t2 = pd.read_csv(filename+'valid_qrel.tsv', sep='\t') # 验证集 正样本
valid_run_t2 = pd.read_csv(filename+'valid_run.tsv', sep='\t', header=None) # 验证样本
valid_run_t2.columns = ['userId','itemIds']
test_run_t2 = pd.read_csv(filename+'test_run.tsv', sep='\t', header=None) # 测试样本
test_run_t2.columns = ['userId','itemIds']
train_t2['source'] = 0
train_5core_t2['rating']*=5
train_5core_t2['source'] = 1
train2 = pd.concat([train_t2,train_5core_t2])


### 对s1,s2,s3的数据过滤
valid_qrel_s1['rating']*=5
valid_qrel_s2['rating']*=5
valid_qrel_s3['rating']*=5

all_s1 = pd.concat([train_s1,valid_qrel_s1])
all_s2 = pd.concat([train_s2,valid_qrel_s2])
all_s3 = pd.concat([train_s3,valid_qrel_s3])

# 这个部分加了验证集的正例和负例
valid_t1 = get_Source(valid_qrel_t1,valid_run_t1)
valid_t2 = get_Source(valid_qrel_t2,valid_run_t2)

# 得到t1,t2测试集
test_t1 = get_init_Test(test_run_t1)
test_t2 = get_init_Test(test_run_t2)

use_t1 = pd.concat([valid_t1,test_t1])
use_t2 = pd.concat([valid_t2,test_t2])

t1_all_items = set(use_t1['itemId'].values)
t2_all_items = set(use_t2['itemId'].values)

def is_in_t1(x):
    if x in t1_all_items:
        return True
    else:
        return False

def is_in_t2(x):
    if x in t2_all_items:
        return True
    else:
        return False

###
all_s1_t1 = all_s1[all_s1['itemId'].apply(lambda x:is_in_t1(x))]
all_s2_t1 = all_s2[all_s2['itemId'].apply(lambda x:is_in_t1(x))]
all_s3_t1 = all_s3[all_s3['itemId'].apply(lambda x:is_in_t1(x))]

###
all_s1_t2 = all_s1[all_s1['itemId'].apply(lambda x:is_in_t2(x))]
all_s2_t2 = all_s2[all_s2['itemId'].apply(lambda x:is_in_t2(x))]
all_s3_t2 = all_s3[all_s3['itemId'].apply(lambda x:is_in_t2(x))]

all_s_t1 = pd.concat([all_s1_t1,all_s2_t1,all_s3_t1])
all_s_t2 = pd.concat([all_s1_t2,all_s2_t2,all_s3_t2])

del all_s_t1['source'], all_s_t2['source']

train1_added = pd.concat([all_s_t1,train1])
train2_added = pd.concat([all_s_t2,train2])

def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)
mkdir('./embedding/')

### embedding特征
re_train = True  # 设置是否重新训练word2vec,此处不重新训练
## 获得embedding特征，前两个是user，后两个是item
filename = './embeddings/'
name1 = filename+'word2vec_t1_users'
name2 = filename+'word2vec_t2_users'
name3 = filename+'word2vec_t1_items'
name4 = filename+'word2vec_t2_items'

userId = 'userId'
itemId = 'itemId'

embedding_t1_users = get_w2v_embedding(train1,name1,'userId','itemId',re_train,16)
embedding_t2_users = get_w2v_embedding(train2,name2,'userId','itemId',re_train,16)

embedding_t1_items = get_w2v_embedding(train1,name3,'itemId','userId',re_train,16)
embedding_t2_items = get_w2v_embedding(train2,name4,'itemId','userId',re_train,16)

name3 = filename+'word2vec_t1_items_test'
name4 = filename+'word2vec_t2_items_test'
embedding_t1_items_test = get_w2v_embedding(pd.concat([train1,valid_qrel_t1]),name3,'itemId','userId',re_train,16)
embedding_t2_items_test = get_w2v_embedding(pd.concat([train2,valid_qrel_t2]),name4,'itemId','userId',re_train,16)

all_s1_t1 = pd.concat([all_s1_t1,train1])
all_s2_t1 = pd.concat([all_s2_t1,train1])
all_s3_t1 = pd.concat([all_s3_t1,train1])

all_s1_t2 = pd.concat([all_s1_t2,train2])
all_s2_t2 = pd.concat([all_s2_t2,train2])
all_s3_t2 = pd.concat([all_s3_t2,train2])

# 分别对s1,s2,s3计算相似度
name3 = filename+'word2vec_t1_items_s1'
name4 = filename+'word2vec_t2_items_s1'
s1_embedding_t1_items = get_w2v_embedding(all_s1_t1,name3,'itemId','userId',re_train,16)
s1_embedding_t2_items = get_w2v_embedding(all_s1_t2,name4,'itemId','userId',re_train,16)

name3 = filename+'word2vec_t1_items_s1_test'
name4 = filename+'word2vec_t2_items_s1_test'
s1_embedding_t1_items_test = get_w2v_embedding(pd.concat([all_s1_t1,valid_qrel_t1]),name3,'itemId','userId',re_train,16)
s1_embedding_t2_items_test = get_w2v_embedding(pd.concat([all_s1_t2,valid_qrel_t2]),name4,'itemId','userId',re_train,16)


name3 = filename+'word2vec_t1_items_s2'
name4 = filename+'word2vec_t2_items_s2'
s2_embedding_t1_items = get_w2v_embedding(all_s2_t1,name3,'itemId','userId',re_train,16)
s2_embedding_t2_items = get_w2v_embedding(all_s2_t2,name4,'itemId','userId',re_train,16)

name3 = filename+'word2vec_t1_items_s2_test'
name4 = filename+'word2vec_t2_items_s2_test'
s2_embedding_t1_items_test = get_w2v_embedding(pd.concat([all_s2_t1,valid_qrel_t1]),name3,'itemId','userId',re_train,16)
s2_embedding_t2_items_test = get_w2v_embedding(pd.concat([all_s2_t2,valid_qrel_t2]),name4,'itemId','userId',re_train,16)


name3 = filename+'word2vec_t1_items_s3'
name4 = filename+'word2vec_t2_items_s3'
s3_embedding_t1_items = get_w2v_embedding(all_s3_t1,name3,'itemId','userId',re_train,16)
s3_embedding_t2_items = get_w2v_embedding(all_s3_t2,name4,'itemId','userId',re_train,16)

name3 = filename+'word2vec_t1_items_s3_test'
name4 = filename+'word2vec_t2_items_s3_test'
s3_embedding_t1_items_test = get_w2v_embedding(pd.concat([all_s3_t1,valid_qrel_t1]),name3,'itemId','userId',re_train,16)
s3_embedding_t2_items_test = get_w2v_embedding(pd.concat([all_s3_t2,valid_qrel_t2]),name4,'itemId','userId',re_train,16)


name3 = filename+'word2vec_t1_items_added'
name4 = filename+'word2vec_t2_items_added'
embedding_t1_items_added = get_w2v_embedding(train1_added,name3,'itemId','userId',re_train,30)
embedding_t2_items_added = get_w2v_embedding(train2_added,name4,'itemId','userId',re_train,30)

name3 = filename+'word2vec_t1_items_added_test'
name4 = filename+'word2vec_t2_items_added_test'
embedding_t1_items_added_test = get_w2v_embedding(pd.concat([train1_added,valid_qrel_t1]),name3,'itemId','userId',re_train,30)
embedding_t2_items_added_test = get_w2v_embedding(pd.concat([train2_added,valid_qrel_t2]),name4,'itemId','userId',re_train,30)

# TF-IDF特征
df_tfidf_emb_t1 = tfidf_svd(train1, f1='itemId', f2='userId', n_components=24)
df_tfidf_emb_t2 = tfidf_svd(train2, f1='itemId', f2='userId', n_components=24)

# 统计特征
feature_list_t1 = get_Statistical_Features(train_t1)
feature_list_t2 = get_Statistical_Features(train_t2)

# 统计特征_added
feature_list_t1_added = get_Statistical_Features_added(train1_added)
feature_list_t2_added = get_Statistical_Features_added(train2_added)

# 加入上述特征
Train1 = get_Train(valid_qrel_t1, valid_run_t1, [feature_list_t1,feature_list_t1_added],[embedding_t1_users,embedding_t1_items,df_tfidf_emb_t1])
Train2 = get_Train(valid_qrel_t2, valid_run_t2,[feature_list_t2,feature_list_t2_added],[embedding_t2_users,embedding_t2_items,df_tfidf_emb_t2])
Test1 = get_Test(test_run_t1, [feature_list_t1,feature_list_t1_added],[embedding_t1_users,embedding_t1_items,df_tfidf_emb_t1])
Test2 = get_Test(test_run_t2, [feature_list_t2,feature_list_t2_added],[embedding_t2_users,embedding_t2_items,df_tfidf_emb_t2])

del feature_list_t1,feature_list_t2,feature_list_t1_added,feature_list_t2_added

############# cos
feature_name = ['cf','cos']
[Train1,Train2,Test1,Test2] = cal_sim_for_all(train1,train2,pd.concat([train1,valid_qrel_t1]),pd.concat([train2,valid_qrel_t2]),\
                        Train1,Train2,Test1,Test2,\
                        embedding_t1_items,embedding_t2_items,embedding_t1_items_test,embedding_t2_items_test,\
                        feature_name)

feature_name = ['all_Source_global_cf','all_Source_global_cos']
[Train1,Train2,Test1,Test2] = cal_sim_for_all(train1_added,train2_added,pd.concat([train1_added,valid_qrel_t1]),pd.concat([train2_added,valid_qrel_t2]),\
                        Train1,Train2,Test1,Test2,\
                        embedding_t1_items_added,embedding_t2_items_added,embedding_t1_items_added_test,embedding_t2_items_added_test,\
                        feature_name)

feature_name = ['s1_cf','s1_cos']
[Train1,Train2,Test1,Test2] = cal_sim_for_all(all_s1_t1,all_s1_t2,pd.concat([all_s1_t1,valid_qrel_t1]),pd.concat([all_s1_t2,valid_qrel_t2]),\
                        Train1,Train2,Test1,Test2,\
                        s1_embedding_t1_items,s1_embedding_t2_items,s1_embedding_t1_items_test,s1_embedding_t2_items_test,\
                        feature_name)

feature_name = ['s2_cf','s2_cos']
[Train1,Train2,Test1,Test2] = cal_sim_for_all(all_s2_t1,all_s2_t2,pd.concat([all_s2_t1,valid_qrel_t1]),pd.concat([all_s2_t2,valid_qrel_t2]),\
                        Train1,Train2,Test1,Test2,\
                        s2_embedding_t1_items,s2_embedding_t2_items,s2_embedding_t1_items_test,s2_embedding_t2_items_test,\
                        feature_name)

feature_name = ['s3_cf','s3_cos']
[Train1,Train2,Test1,Test2] = cal_sim_for_all(all_s3_t1,all_s3_t2,pd.concat([all_s3_t1,valid_qrel_t1]),pd.concat([all_s3_t2,valid_qrel_t2]),\
                        Train1,Train2,Test1,Test2,\
                        s3_embedding_t1_items,s3_embedding_t2_items,s3_embedding_t1_items_test,s3_embedding_t2_items_test,\
                        feature_name)


# LightGCN stacking
nn_valid_t1_df = pd.read_csv('./merge/t1/LightGCN_with_w2v_valid.tsv', sep='\t')
nn_test_t1_df = pd.read_csv('./merge/t1/LightGCN_with_w2v_test.tsv', sep='\t')

nn_valid_t2_df = pd.read_csv('./merge/t2/LightGCN_with_w2v_valid.tsv', sep='\t')
nn_test_t2_df = pd.read_csv('./merge/t2/LightGCN_with_w2v_test.tsv', sep='\t')

###
Train1 = Train1.merge(nn_valid_t1_df, on=['userId','itemId'], how='left')
Test1  = Test1.merge(nn_test_t1_df, on=['userId','itemId'], how='left')

Train2 = Train2.merge(nn_valid_t2_df, on=['userId','itemId'], how='left')
Test2  = Test2.merge(nn_test_t2_df, on=['userId','itemId'], how='left')

# LightGCN stacking
nn_valid_t1_df = pd.read_csv('./merge/t1/LightGCN_without_w2v_valid.tsv', sep='\t')
nn_test_t1_df = pd.read_csv('./merge/t1/LightGCN_without_w2v_test.tsv', sep='\t')

nn_valid_t2_df = pd.read_csv('./merge/t2/LightGCN_without_w2v_valid.tsv', sep='\t')
nn_test_t2_df = pd.read_csv('./merge/t2/LightGCN_without_w2v_test.tsv', sep='\t')

###
Train1 = Train1.merge(nn_valid_t1_df, on=['userId','itemId'], how='left')
Test1  = Test1.merge(nn_test_t1_df, on=['userId','itemId'], how='left')

Train2 = Train2.merge(nn_valid_t2_df, on=['userId','itemId'], how='left')
Test2  = Test2.merge(nn_test_t2_df, on=['userId','itemId'], how='left')


def lgb_ranker_model(train,test,k_fold=5,seed=2022):
    trn_df = train
    user_set = get_kfold_users(trn_df, n=k_fold,seed=seed)
    score_list = []
    sub_preds = np.zeros(test.shape[0])
    
    for n_fold, valid_user in enumerate(user_set):
        print('************************************ {} ************************************'.format(str(n_fold+1)))
        train_idx = trn_df[~trn_df['userId'].isin(valid_user)] # add slide user
        valid_idx = trn_df[trn_df['userId'].isin(valid_user)]

        # 训练集与验证集的用户分组
        train_idx.sort_values(by=['userId'], inplace=True)
        g_train = train_idx.groupby(['userId'], as_index=False).count()["label"].values

        valid_idx.sort_values(by=['userId'], inplace=True)
        g_val = valid_idx.groupby(['userId'], as_index=False).count()["label"].values
        
        # 训练模型
        lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=7, reg_alpha=0.0, reg_lambda=1,
                                    max_depth=-1, n_estimators=1000, subsample=0.8, colsample_bytree=0.8, 
                                    subsample_freq=1,learning_rate=0.03, min_child_weight=10, random_state=2022, 
                                    n_jobs=-1)  
 
        feats = [f for f in train_idx.columns if f not in ['userId','itemId','label']]
        lgb_ranker.fit(train_idx[feats], train_idx['label'], group=g_train,
                       eval_set=[(valid_idx[feats], valid_idx['label'])], eval_group= [g_val], 
                       eval_at=[10], eval_metric=['ndcg', ], early_stopping_rounds=200, verbose=100)
        
        # 预测验证集结果
        valid_idx['pred_score'] = lgb_ranker.predict(valid_idx[feats], num_iteration=lgb_ranker.best_iteration_)
    
        # 对输出结果进行归一化
        valid_idx['pred_score'] = valid_idx[['pred_score']].transform(lambda x: norm_sim(x))
        valid_idx.sort_values(by=['userId', 'pred_score'])
        valid_idx['pred_rank'] = valid_idx.groupby(['userId'])['pred_score'].rank(ascending=False, method='first')

        # 将验证集的预测结果放到一个列表中，后面进行拼接
        score_list.append(valid_idx[['userId', 'itemId', 'pred_score', 'pred_rank']])
        
        # 测试
        sub_preds += lgb_ranker.predict(test[feats], lgb_ranker.best_iteration_)
    
    test['pred_score'] = sub_preds / k_fold
    test['pred_score'] = test['pred_score'].transform(lambda x: norm_sim(x))
    test.sort_values(by=['userId', 'pred_score'])
    test['pred_rank'] = test.groupby(['userId'])['pred_score'].rank(ascending=False, method='first')
    return score_list,test

##### 5fold
print('====================================== t1 ======================================')
score_list_t1,test_t1_pred = lgb_ranker_model(Train1,Test1,5,2022)
print('====================================== t2 ======================================')
score_list_t2,test_t2_pred = lgb_ranker_model(Train2,Test2,5,2022)

### t1
score_df_t1 = pd.concat(score_list_t1, axis=0)
output_cols = ['userId','itemId','pred_score']

score_df_t1 = score_df_t1[output_cols].rename(columns={'pred_score':'score'})
test_t1_pred = test_t1_pred[output_cols].rename(columns={'pred_score':'score'})
score_df_t1 = score_df_t1.sort_values(by=['userId','score'],ascending=[True,False])
test_t1_pred = test_t1_pred.sort_values(by=['userId','score'],ascending=[True,False])

valid_qrel_t1 = valid_qrel_t1.sort_values(by=['userId'],ascending=[True])

### t2
score_df_t2 = pd.concat(score_list_t2, axis=0)
output_cols = ['userId','itemId','pred_score']

score_df_t2 = score_df_t2[output_cols].rename(columns={'pred_score':'score'})
test_t2_pred = test_t2_pred[output_cols].rename(columns={'pred_score':'score'})
score_df_t2 = score_df_t2.sort_values(by=['userId','score'],ascending=[True,False])
test_t2_pred = test_t2_pred.sort_values(by=['userId','score'],ascending=[True,False])

valid_qrel_t2 = valid_qrel_t2.sort_values(by=['userId'],ascending=[True])

output_dir = './merge/result_rank1/'
score_df_t1.to_csv(output_dir+'t1/valid_pred.tsv',index=False,sep='\t')
test_t1_pred.to_csv(output_dir+'t1/test_pred.tsv',index=False,sep='\t')
score_df_t2.to_csv(output_dir+'t2/valid_pred.tsv',index=False,sep='\t')
test_t2_pred.to_csv(output_dir+'t2/test_pred.tsv',index=False,sep='\t')


##### 10fold
print('====================================== t1 ======================================')
score_list_t1,test_t1_pred = lgb_ranker_model(Train1,Test1,10,2021)
print('====================================== t2 ======================================')
score_list_t2,test_t2_pred = lgb_ranker_model(Train2,Test2,10,2021)

### t1
score_df_t1 = pd.concat(score_list_t1, axis=0)
output_cols = ['userId','itemId','pred_score']

score_df_t1 = score_df_t1[output_cols].rename(columns={'pred_score':'score'})
test_t1_pred = test_t1_pred[output_cols].rename(columns={'pred_score':'score'})
score_df_t1 = score_df_t1.sort_values(by=['userId','score'],ascending=[True,False])
test_t1_pred = test_t1_pred.sort_values(by=['userId','score'],ascending=[True,False])

valid_qrel_t1 = valid_qrel_t1.sort_values(by=['userId'],ascending=[True])

### t2
score_df_t2 = pd.concat(score_list_t2, axis=0)
output_cols = ['userId','itemId','pred_score']

score_df_t2 = score_df_t2[output_cols].rename(columns={'pred_score':'score'})
test_t2_pred = test_t2_pred[output_cols].rename(columns={'pred_score':'score'})
score_df_t2 = score_df_t2.sort_values(by=['userId','score'],ascending=[True,False])
test_t2_pred = test_t2_pred.sort_values(by=['userId','score'],ascending=[True,False])

valid_qrel_t2 = valid_qrel_t2.sort_values(by=['userId'],ascending=[True])

output_dir = './merge/result_rank2/'
score_df_t1.to_csv(output_dir+'t1/valid_pred.tsv',index=False,sep='\t')
test_t1_pred.to_csv(output_dir+'t1/test_pred.tsv',index=False,sep='\t')
score_df_t2.to_csv(output_dir+'t2/valid_pred.tsv',index=False,sep='\t')
test_t2_pred.to_csv(output_dir+'t2/test_pred.tsv',index=False,sep='\t')