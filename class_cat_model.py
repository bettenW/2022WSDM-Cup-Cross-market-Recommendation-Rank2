import pandas as pd
import os
import gc
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, BayesianRidge
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from scipy.sparse.linalg import svds
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from tools import *

train_s1 = pd.read_csv('DATA/s1/train.tsv', sep='\t')
train_5core_s1 = pd.read_csv('DATA/s1/train_5core.tsv', sep='\t')
valid_qrel_s1 = pd.read_csv('DATA/s1/valid_qrel.tsv', sep='\t') # 验证集 正样本
valid_run_s1 = pd.read_csv('DATA/s1/valid_run.tsv', sep='\t', header=None) # 验证样本
valid_run_s1.columns = ['userId','itemIds']

train_s2 = pd.read_csv('DATA/s2/train.tsv', sep='\t')
train_5core_s2 = pd.read_csv('DATA/s2/train_5core.tsv', sep='\t')
valid_qrel_s2 = pd.read_csv('DATA/s2/valid_qrel.tsv', sep='\t')
valid_run_s2 = pd.read_csv('DATA/s2/valid_run.tsv', sep='\t', header=None)
valid_run_s2.columns = ['userId','itemIds']

train_s3 = pd.read_csv('DATA/s3/train.tsv', sep='\t')
train_5core_s3 = pd.read_csv('DATA/s3/train_5core.tsv', sep='\t')
valid_qrel_s3 = pd.read_csv('DATA/s3/valid_qrel.tsv', sep='\t')
valid_run_s3 = pd.read_csv('DATA/s3/valid_run.tsv', sep='\t', header=None)
valid_run_s3.columns = ['userId','itemIds']

train_t1 = pd.read_csv('DATA/t1/train.tsv', sep='\t')
train_5core_t1 = pd.read_csv('DATA/t1/train_5core.tsv', sep='\t')
valid_qrel_t1 = pd.read_csv('DATA/t1/valid_qrel.tsv', sep='\t') # 验证集 正样本
valid_run_t1 = pd.read_csv('DATA/t1/valid_run.tsv', sep='\t', header=None) # 验证样本
valid_run_t1.columns = ['userId','itemIds']
test_run_t1 = pd.read_csv('DATA/t1/test_run.tsv', sep='\t', header=None) # 测试样本
test_run_t1.columns = ['userId','itemIds']

train_t2 = pd.read_csv('DATA/t2/train.tsv', sep='\t')
train_5core_t2 = pd.read_csv('DATA/t2/train_5core.tsv', sep='\t')
valid_qrel_t2 = pd.read_csv('DATA/t2/valid_qrel.tsv', sep='\t') # 验证集 正样本
valid_run_t2 = pd.read_csv('DATA/t2/valid_run.tsv', sep='\t', header=None) # 验证样本
valid_run_t2.columns = ['userId','itemIds']
test_run_t2 = pd.read_csv('DATA/t2/test_run.tsv', sep='\t', header=None) # 测试样本
test_run_t2.columns = ['userId','itemIds']

# 构建训练数据和测试数据
#### t1
valid_t1_df = []
for items in valid_run_t1[['userId','itemIds']].values:
    uid = items[0]
    its = items[1].split(',')
    for i, it in enumerate(its):
        valid_t1_df.append([uid,it])
valid_t1_df = pd.DataFrame(valid_t1_df)
valid_t1_df.columns = ['userId','itemId']
valid_t1_df = valid_t1_df.merge(valid_qrel_t1, on=['userId','itemId'], how='left')
valid_t1_df.columns = ['userId','itemId','label']
valid_t1_df['label'] = valid_t1_df['label'].fillna(0)

test_t1_df = []
for items in test_run_t1[['userId','itemIds']].values:
    uid = items[0]
    its = items[1].split(',')
    for i, it in enumerate(its):
        test_t1_df.append([uid,it])
test_t1_df = pd.DataFrame(test_t1_df)
test_t1_df.columns = ['userId','itemId']
        
#### t2
valid_t2_df = []
for items in valid_run_t2[['userId','itemIds']].values:
    uid = items[0]
    its = items[1].split(',')
    for i, it in enumerate(its):
        valid_t2_df.append([uid,it])
valid_t2_df = pd.DataFrame(valid_t2_df)
valid_t2_df.columns = ['userId','itemId']
valid_t2_df = valid_t2_df.merge(valid_qrel_t2, on=['userId','itemId'], how='left')
valid_t2_df.columns = ['userId','itemId','label']
valid_t2_df['label'] = valid_t2_df['label'].fillna(0)

test_t2_df = []
for items in test_run_t2[['userId','itemIds']].values:
    uid = items[0]
    its = items[1].split(',')
    for i, it in enumerate(its):
        test_t2_df.append([uid,it])
test_t2_df = pd.DataFrame(test_t2_df)
test_t2_df.columns = ['userId','itemId']

# 合并数据集
data_t1_df = pd.concat([valid_t1_df,test_t1_df], axis=0, ignore_index=True)
data_t2_df = pd.concat([valid_t2_df,test_t2_df], axis=0, ignore_index=True)


def base_feature(df1_, df2_, df3_, flag):
    df1 = df1_.copy() # train_t1
    df2 = df2_.copy() # data_t1_df
    df3 = df3_.copy() # train_5core_t1
    
    print('count/nunique...')
    # train_t1
    userId_feat = df1.groupby(['userId'])['itemId'].agg({'count','nunique'}).reset_index()
    userId_feat.columns = ['userId','userId_cnts','userId_itemId_nunique']
    
    itemId_feat = df1.groupby(['itemId'])['userId'].agg({'count','nunique'}).reset_index()
    itemId_feat.columns = ['itemId','itemId_cnts','itemId_userId_nunique']
    
    df2 = df2.merge(userId_feat, on='userId', how='left')
    df2 = df2.merge(itemId_feat, on='itemId', how='left')
    
    # train_5core_t1
    userId_feat = df3.groupby(['userId'])['itemId'].agg({'count','nunique'}).reset_index()
    userId_feat.columns = ['userId','userId_cnts','userId_itemId_nunique']
    
    itemId_feat = df3.groupby(['itemId'])['userId'].agg({'count','nunique'}).reset_index()
    itemId_feat.columns = ['itemId','itemId_cnts','itemId_userId_nunique']
    
    df2 = df2.merge(userId_feat, on='userId', how='left')
    df2 = df2.merge(itemId_feat, on='itemId', how='left')
    
    print('rating feat...')
    userId_feat = df1.groupby(['userId'])['rating'].agg({'mean'}).reset_index()
    userId_feat.columns = ['userId','userId_rating_mean']
    
    itemId_feat = df1.groupby(['itemId'])['rating'].agg({'mean'}).reset_index()
    itemId_feat.columns = ['itemId','itemId_rating_mean']
    
    df2 = df2.merge(userId_feat, on='userId', how='left')
    df2 = df2.merge(itemId_feat, on='itemId', how='left')
    
    print('word2vec...')
    df2 = df2.merge(emb_feature(df1, 'userId', 'itemId'), on='userId', how='left')
    df2 = df2.merge(emb_feature(df1, 'itemId', 'userId'), on='itemId', how='left')
    
    df2 = df2.merge(emb_feature(df3, 'userId', 'itemId'), on='userId', how='left')
    df2 = df2.merge(emb_feature(df3, 'itemId', 'userId'), on='itemId', how='left')
    
    print('word2vec similar feat...')
    vec = emb_feature2(df1, 'userId', 'itemId')
    
    tmp = df2[['userId','itemId']].drop_duplicates(subset=['userId','itemId'], keep='last')
    tmp = tmp.merge(vec, on='itemId', how='left')
    
    # 聚合历史交互itemId
    tmp1 = df1.merge(vec, on='itemId', how='left')
    tmp1 = tmp1.groupby(['userId'])['itemId_w2v_emb'].agg({list}).reset_index()
    tmp1.columns = ['userId','emb_list']
    
    tmp = tmp.merge(tmp1, on='userId', how='left')
    
    tmp['similar_itemId_mean'] =tmp.apply(lambda index: similar_get(index['itemId_w2v_emb'], index['emb_list']), axis=1)
    
    tmp['similar_itemId_max']    = tmp['similar_itemId_mean'].apply(lambda x: x[1])
    tmp['similar_itemId_min']    = tmp['similar_itemId_mean'].apply(lambda x: x[2])
    tmp['similar_itemId_std']    = tmp['similar_itemId_mean'].apply(lambda x: x[3])
    tmp['similar_itemId_sum']    = tmp['similar_itemId_mean'].apply(lambda x: x[4])
    tmp['similar_itemId_median'] = tmp['similar_itemId_mean'].apply(lambda x: x[5])
    tmp['similar_itemId_percentile_25']   = tmp['similar_itemId_mean'].apply(lambda x: x[6])
    tmp['similar_itemId_percentile_75']   = tmp['similar_itemId_mean'].apply(lambda x: x[7])
    tmp['similar_itemId_percentile_5']   = tmp['similar_itemId_mean'].apply(lambda x: x[8])
    tmp['similar_itemId_percentile_95']   = tmp['similar_itemId_mean'].apply(lambda x: x[9])
    tmp['similar_itemId_mean']   = tmp['similar_itemId_mean'].apply(lambda x: x[0])
    
    del tmp['itemId_w2v_emb'], tmp['emb_list']
    df2  = df2.merge(tmp, on=['userId','itemId'], how='left')

    ### embedding2
    for i in range(32):
        vec['itemId_w2v_emb_'+str(i)] = vec['itemId_w2v_emb'].apply(lambda x:x[i])
    del vec['itemId_w2v_emb']
    df2  = df2.merge(vec, on='itemId', how='left')

    df2 = reduce_mem_usage(df2)
    
    if flag == 't1':
        ### new dota feature
        df3['rating'] = df3['rating'].values * 5
        df_all = pd.concat([df1, df3], axis=0, ignore_index=True)
        df_all['ui'] = df_all['userId'] + '_' + df_all['itemId']

        user_feats = user_features(df_all, df_all)
        item_feats = item_features(df_all, df_all)
        ui_feats = df_all[["ui"]].drop_duplicates().reset_index(drop=True)
        ui_feats['ui_cnt'] = ui_feats["ui"].map(df_all['ui'].value_counts())
        ui_feats['ui_rsum'] = ui_feats["ui"].map(df_all.groupby('ui')['rating'].sum())
        ui_feats['ui_rmean'] = ui_feats["ui"].map(df_all.groupby('ui')['rating'].mean())
        ui_embs = embs_ui_scores(user_feats, item_feats)

        df2['ui'] = df2['userId'] + '_' + df2['itemId']
        df2 = df2.merge(ui_embs, on='ui', how='left')
        df2 = df2.merge(user_feats, on='userId', how='left')
        df2 = df2.merge(item_feats, on='itemId', how='left')
        del df2['ui']
    
    return df2

print('====================================== t1 ======================================')
data_t1_df = base_feature(train_t1, data_t1_df, train_5core_t1, 't1')
print('====================================== t2 ======================================')
data_t2_df = base_feature(train_t2, data_t2_df, train_5core_t2, 't2')


valid_t1_df = data_t1_df[~data_t1_df.label.isnull()].reset_index(drop=True)
test_t1_df  = data_t1_df[data_t1_df.label.isnull()].reset_index(drop=True)

valid_t2_df = data_t2_df[~data_t2_df.label.isnull()].reset_index(drop=True)
test_t2_df  = data_t2_df[data_t2_df.label.isnull()].reset_index(drop=True)

data_t1_df = data_t1_df[['userId','itemId']]
data_t2_df = data_t2_df[['userId','itemId']]
gc.collect()

for i in range(0,10):
    print('====================================== {} ======================================'.format(str(i)))
    recom_valid_t1_df = pd.read_csv('./recall_feature_data/recom_valid_t1_feat{}.csv'.format(str(i)))
    recom_test_t1_df = pd.read_csv('./recall_feature_data/recom_test_t1_feat{}.csv'.format(str(i)))

    recom_valid_t2_df = pd.read_csv('./recall_feature_data/recom_valid_t2_feat{}.csv'.format(str(i)))
    recom_test_t2_df = pd.read_csv('./recall_feature_data/recom_test_t2_feat{}.csv'.format(str(i)))
    print(recom_valid_t1_df.columns)
    # merge
    valid_t1_df = valid_t1_df.merge(recom_valid_t1_df, on=['userId','itemId'], how='left')
    test_t1_df = test_t1_df.merge(recom_test_t1_df, on=['userId','itemId'], how='left')

    valid_t2_df = valid_t2_df.merge(recom_valid_t2_df, on=['userId','itemId'], how='left')
    test_t2_df = test_t2_df.merge(recom_test_t2_df, on=['userId','itemId'], how='left')
    
    if i < 4:
        print('====================================== swing {} ======================================'.format(str(i)))
        recom_valid_t1_df = pd.read_csv('./recall_feature_data/recom_valid_t1_swing_feat{}.csv'.format(str(i)))
        recom_test_t1_df = pd.read_csv('./recall_feature_data/recom_test_t1_swing_feat{}.csv'.format(str(i)))

        recom_valid_t2_df = pd.read_csv('./recall_feature_data/recom_valid_t2_swing_feat{}.csv'.format(str(i)))
        recom_test_t2_df = pd.read_csv('./recall_feature_data/recom_test_t2_swing_feat{}.csv'.format(str(i)))
        print(recom_valid_t1_df.columns)
        # merge
        valid_t1_df = valid_t1_df.merge(recom_valid_t1_df, on=['userId','itemId'], how='left')
        test_t1_df = test_t1_df.merge(recom_test_t1_df, on=['userId','itemId'], how='left')

        valid_t2_df = valid_t2_df.merge(recom_valid_t2_df, on=['userId','itemId'], how='left')
        test_t2_df = test_t2_df.merge(recom_test_t2_df, on=['userId','itemId'], how='left')


# LightGCN stacking
nn_valid_t1_df = pd.read_csv('./merge/t1/LightGCN_with_w2v_valid.tsv', sep='\t')
nn_test_t1_df = pd.read_csv('./merge/t1/LightGCN_with_w2v_test.tsv', sep='\t')

nn_valid_t2_df = pd.read_csv('./merge/t2/LightGCN_with_w2v_valid.tsv', sep='\t')
nn_test_t2_df = pd.read_csv('./merge/t2/LightGCN_with_w2v_test.tsv', sep='\t')

###
valid_t1_df = valid_t1_df.merge(nn_valid_t1_df, on=['userId','itemId'], how='left')
test_t1_df = test_t1_df.merge(nn_test_t1_df, on=['userId','itemId'], how='left')

valid_t2_df = valid_t2_df.merge(nn_valid_t2_df, on=['userId','itemId'], how='left')
test_t2_df = test_t2_df.merge(nn_test_t2_df, on=['userId','itemId'], how='left')

# LightGCN stacking
nn_valid_t1_df = pd.read_csv('./merge/t1/LightGCN_without_w2v_valid.tsv', sep='\t')
nn_test_t1_df = pd.read_csv('./merge/t1/LightGCN_without_w2v_test.tsv', sep='\t')

nn_valid_t2_df = pd.read_csv('./merge/t2/LightGCN_without_w2v_valid.tsv', sep='\t')
nn_test_t2_df = pd.read_csv('./merge/t2/LightGCN_without_w2v_test.tsv', sep='\t')

###
valid_t1_df = valid_t1_df.merge(nn_valid_t1_df, on=['userId','itemId'], how='left')
test_t1_df = test_t1_df.merge(nn_test_t1_df, on=['userId','itemId'], how='left')

valid_t2_df = valid_t2_df.merge(nn_valid_t2_df, on=['userId','itemId'], how='left')
test_t2_df = test_t2_df.merge(nn_test_t2_df, on=['userId','itemId'], how='left')


for col in ['userId','itemId']:
    valid_t1_df[col] = label_encode(data_t1_df[col], valid_t1_df[col])
    test_t1_df[col] = label_encode(data_t1_df[col], test_t1_df[col])

    valid_t2_df[col] = label_encode(data_t2_df[col], valid_t2_df[col])
    test_t2_df[col] = label_encode(data_t2_df[col], test_t2_df[col])
    
def cv_model(clf, train_x, train_y, test_x, clf_name, seed=2021):
    
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} {}************************************'.format(str(i+1), str(seed)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
               
        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'min_child_samples': 10,
                'num_leaves': 2**3-1,
                'lambda_l2': 10,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'learning_rate': 0.03,
                'seed': seed,
                'n_jobs':-1,
                'verbose': -1,
            }

            model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], 
                              categorical_feature=[], verbose_eval=200, early_stopping_rounds=200)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)
               
        if clf_name == "xgb":
            train_matrix = clf.DMatrix(trn_x , label=trn_y)
            valid_matrix = clf.DMatrix(val_x , label=val_y)
            test_matrix = clf.DMatrix(test_x)
            
            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'auc',
                      'gamma': 1,
                      'min_child_weight': 5,
                      'max_depth': 3,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.8,
                      'colsample_bylevel': 0.8,
                      'eta': 0.04,
                      'tree_method': 'exact',
                      'seed': seed,
                      'n_jobs': -1
                      }
            
            watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
            
            model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=200, early_stopping_rounds=200)
            val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
            test_pred = model.predict(test_matrix , ntree_limit=model.best_ntree_limit)
                 
        if clf_name == "cat":
            params = {'learning_rate': 0.02, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type':'Bernoulli','random_seed':seed,
                      'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}
            
            model = clf(iterations=20000, **params)
            model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                      cat_features=[], use_best_model=True, verbose=500)
            
            val_pred  = model.predict(val_x)
            test_pred = model.predict(test_x)
            
        train[valid_index] = val_pred
        test += test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))
        
        print(cv_scores)
       
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test

def cat_model(x_train, y_train, x_test, seed):
    cat_train, cat_test = cv_model(CatBoostRegressor, x_train, y_train, x_test, "cat", seed) 
    return cat_train, cat_test

              
random_list = [1,10,100,1000,2021]
print('====================================== t1 ======================================')
cols = [f for f in valid_t1_df.columns if f not in ['label']]
lgb_train_t1 = np.zeros(valid_t1_df.shape[0])
lgb_test_t1 = np.zeros(test_t1_df.shape[0])
for sd in random_list:
    lgb_train, lgb_test = cat_model(valid_t1_df[cols], valid_t1_df['label'].astype(int), test_t1_df[cols], sd)
    lgb_train_t1 += lgb_train / len(random_list)
    lgb_test_t1  += lgb_test  / len(random_list)
              
print('====================================== t2 ======================================')
cols = [f for f in valid_t2_df.columns if f not in ['label']]
lgb_train_t2 = np.zeros(valid_t2_df.shape[0])
lgb_test_t2 = np.zeros(test_t2_df.shape[0])
for sd in random_list:
    lgb_train, lgb_test = cat_model(valid_t2_df[cols], valid_t2_df['label'].astype(int), test_t2_df[cols], sd)
    lgb_train_t2 += lgb_train / len(random_list)
    lgb_test_t2  += lgb_test  / len(random_list)

for col in ['userId','itemId']:
    valid_t1_df[col] = label_encode2(data_t1_df[col], valid_t1_df[col])
    test_t1_df[col] = label_encode2(data_t1_df[col], test_t1_df[col])

    valid_t2_df[col] = label_encode2(data_t2_df[col], valid_t2_df[col])
    test_t2_df[col] = label_encode2(data_t2_df[col], test_t2_df[col])
    
valid_t1_df['score'] = lgb_train_t1
valid_t1_df = valid_t1_df.sort_values(['userId','score'], ascending=False)
recom_t1_df = valid_t1_df.groupby(['userId'])['itemId'].agg(list).reset_index()
recom_t1_df.columns = ['userId','pred_itemIds']

valid_t2_df['score'] = lgb_train_t2
valid_t2_df = valid_t2_df.sort_values(['userId','score'], ascending=False)
recom_t2_df = valid_t2_df.groupby(['userId'])['itemId'].agg(list).reset_index()
recom_t2_df.columns = ['userId','pred_itemIds']

def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)
mkdir('./merge/result_cat/t1/')
mkdir('./merge/result_cat/t2/')
              
#### 提交结果
### t1
test_t1_df['score'] = lgb_test_t1
test_t1_df = test_t1_df.sort_values(['userId','score'], ascending=False)
test_t1_df[['userId','itemId','score']].to_csv('./merge/result_cat/t1/test_pred.tsv', sep='\t', index=False) 

valid_t1_df[['userId','itemId','score']].to_csv('./merge/result_cat/t1/valid_pred.tsv', sep='\t', index=False)

### t2
test_t2_df['score'] = lgb_test_t2
test_t2_df = test_t2_df.sort_values(['userId','score'], ascending=False)
test_t2_df[['userId','itemId','score']].to_csv('./merge/result_cat/t2/test_pred.tsv', sep='\t', index=False)

valid_t2_df[['userId','itemId','score']].to_csv('./merge/result_cat/t2/valid_pred.tsv', sep='\t', index=False)