# -*- coding: utf-8 -*-
import pandas as pd
import os
import gc
import copy
import networkx as nx
from gensim.models import Word2Vec
import math
import numpy as np
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import random
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import LabelEncoder
import time
import warnings
warnings.filterwarnings('ignore')

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def label_encode(series, series2):
    unique = list(series.unique())
    return series2.map(dict(zip(
        unique, range(series.nunique())
    )))

def label_encode2(series, series2):
    unique = list(series.unique())
    return series2.map(dict(zip(
        range(series.nunique()), unique
    )))

def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)

def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    #idcg = getDCG(relevance)
    idcg = 1
    
    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg

def emb_feature(df, f1, f2):
    emb_size = 32
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, size=emb_size, window=100, min_count=1, sg=1, hs=1, workers=16, seed=2021)
    emb_matrix = []
    print('for seq in sentences:')
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model:
                vec.append(model[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)

    df_emb = pd.DataFrame(emb_matrix)
    df_emb.columns = ['{}_{}_emb_{}'.format(f1, f2, i) for i in range(emb_size)]

    tmp = pd.concat([tmp, df_emb], axis=1)
    del model, emb_matrix, sentences
    gc.collect()
    
    return tmp

def emb_feature2(df, f1, f2):
    emb_size = 32
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, size=emb_size, window=100, min_count=1, sg=1, hs=1, workers=16, seed=2021)
    # 保存结果
    values=set(df[f2].values)
    w2v=[]
    for v in values:
        if v in model:
            a = [v, model[v]]
            w2v.append(a)
    out_df=pd.DataFrame(w2v)
    
    out_df.columns = [f2, f2+'_w2v_emb']
    
    return out_df

def emb_feature3(df, f1, f2):
    
    emb_size = 32
    print('====================================== {} {} ======================================'.format(f1, f2))
    
    G = nx.from_pandas_edgelist(df, f1, f2, edge_attr=True, create_using=nx.Graph())
    
    def get_randomwalk(node, path_length):
        random_walk = [node]
        for i in range(path_length - 1):
            temp = list(G.neighbors(node))
            temp = list(set(temp) - set(random_walk))
            if len(temp) == 0:
                break
            random_node = random.choice(temp)
            random_walk.append(random_node)
            node = random_node
        return random_walk

    all_nodes = list(G.nodes())

    random_walks = []
    for n in all_nodes:
        for i in range(5):
            random_walks.append(get_randomwalk(n, 10))
    
    model = Word2Vec(random_walks, size=emb_size, window=100, min_count=1, sg=1, hs=1, workers=16, seed=2021)
    
    # user
    user_values=set(df[f1].values)
    w2v=[]
    for v in user_values:
        if v in model:
            a = [v, model[v]]
            w2v.append(a)
    user_df = pd.DataFrame(w2v)
    user_df.columns = [f1, f1+'_dw_emb']
    
    # item
    item_values=set(df[f2].values)
    w2v=[]
    for v in item_values:
        if v in model:
            a = [v, model[v]]
            w2v.append(a)
    item_df = pd.DataFrame(w2v)
    item_df.columns = [f2, f2+'_dw_emb']
    
    return user_df, item_df

def w2v_embedding(df, mk, gpk, flag='', vsize=24):
    gpk_keys = df[gpk].unique().tolist() 
    df_mk_gpk_list = df.groupby([mk])[gpk].apply(lambda x: list(x)).reset_index() 
    gpk_list = df_mk_gpk_list[gpk].values.tolist() 
    model = Word2Vec(gpk_list, size=vsize, sg=1, window=5, seed=2009, workers=36, min_count=1)
 
    gpk_dic = []
    gpk_matrix = np.zeros([len(gpk_keys), vsize])
    for i,k in enumerate(tqdm(gpk_keys)):
        try:
            gpk_matrix[i,:] = model.wv[k]
        except:
            pass

    w2v_embs = df[[gpk]].drop_duplicates().reset_index(drop=True)
    for i in range(vsize):
        w2v_embs['{}_{}_w2v_{}_{}'.format(mk,gpk,str(i),flag)] = gpk_matrix[:,i]
 
    return w2v_embs

def tfidf_svd_(data, mk, gpk, n_components = 24): 
    tmp = data.groupby([mk])[gpk].agg(list).reset_index()
    tmp.columns= [mk, 'seqs'] 
    tmp['seqs'] = tmp['seqs'].map(lambda x: ' '.join(x))  
    tfidf = TfidfVectorizer(max_df = 0.9, min_df = 1, sublinear_tf = True)
    res = tfidf.fit_transform(tmp['seqs'])
    svd = TruncatedSVD(n_components = n_components, random_state = 2009)
    
    svd_res = svd.fit_transform(res)
    for i in range(n_components):
        tmp['{}_{}_tfidf_svd_{}'.format(mk,gpk,str(i))] = svd_res[:,i]
    del tmp['seqs']
    gc.collect()
    return tmp

def user_features(df, train_all):
    user_feats = df[["userId"]].drop_duplicates().reset_index(drop=True)
    user_feats['u_cnt'] = user_feats['userId'].map(df['userId'].value_counts())
    user_feats['ui_nunique'] = user_feats['userId'].map(df.groupby('userId')['itemId'].nunique())
    user_feats['ui_rcnt'] = user_feats['userId'].map(df.loc[df['rating'].isnull() == False].groupby('userId')['itemId'].nunique())

    for i in range(0,6):
        if i == 0:
            df_w2v = w2v_embedding(df, 'itemId', 'userId', 'dota', 24)
            user_feats = user_feats.merge(df_w2v, on='userId', how='left')
        else:
            df_w2v = w2v_embedding(df.loc[df['rating'] == i], 'itemId', 'userId', str(i), 24)
            user_feats = user_feats.merge(df_w2v, on='userId', how='left')

    tfidf_embeds = tfidf_svd_(df, 'userId', 'itemId', 24)
    user_feats = user_feats.merge(tfidf_embeds, on='userId', how='left') 
    return user_feats

def item_features(df, train_all):
    item_feats = df[["itemId"]].drop_duplicates().reset_index(drop=True)
    
    item_feats['i_cnt'] =  item_feats['itemId'].map(df['itemId'].value_counts())
    item_feats['iu_nunique'] =  item_feats['itemId'].map(df.groupby('itemId')['userId'].nunique())
    item_feats['iu_rcnt'] =  item_feats['itemId'].map(df.loc[df['rating'].isnull() == False].groupby('itemId')['userId'].nunique())
    
    for i in range(0,6):
        if i == 0:
            df_w2v_rating = w2v_embedding(df = df, mk='userId', gpk='itemId', flag='dota' )
            item_feats = item_feats.merge(df_w2v_rating, on = 'itemId', how = 'left')
        else:
            df_w2v_rating = w2v_embedding(df = df.loc[df['rating'] == i], mk = 'userId', gpk = 'itemId', flag = str(i))
            item_feats = item_feats.merge(df_w2v_rating, on = 'itemId', how = 'left') 
     
    tfidf_embeds = tfidf_svd_(df, mk = 'itemId', gpk = 'userId', n_components = 24)
    item_feats = item_feats.merge(tfidf_embeds, on = 'itemId', how = 'left') 
    return item_feats

def similar_get(x,ys):
    if x is np.nan or ys is np.nan:
        return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
    else:
        score = []
        for y in ys:
            aa = np.dot(x,y)
            denom = np.linalg.norm(x, axis=-1) * np.linalg.norm(y)
            aa = aa / denom
            score.append(aa)
        return [np.mean(score),np.max(score),np.min(score),np.std(score),np.sum(score),np.median(score),np.percentile(score, 25),np.percentile(score, 75),np.percentile(score, 5),np.percentile(score, 95)]

def dot_uv(U,V): 
    return np.dot(U,np.transpose(V))

def cos_uv(U,V):
    norm = np.linalg.norm(U, axis=1).reshape(-1, 1) * np.linalg.norm(V, axis=1) 
    return dot_uv(U,V) / (norm + 1e-9)
    
def eud_uv(U,V):
    UV = dot_uv(U,V)
    UU = np.sum(U * U,axis=1).reshape(-1,1) 
    UU = np.repeat(UU,V.shape[0],axis=1)
    
    VT = np.transpose(V)
    VV = np.sum(VT * VT,axis=0).reshape(1,-1) 
    VV = np.repeat(VV,U.shape[0],axis=0)
    
    return UU - 2*UV + VV 
    
def dota_ui_score(dota_user, dota_item, ucols, icols, flag = 'w2v'):
    dot_ui=dot_uv(dota_user[ucols].values, dota_item[icols].values) 
    cos_ui=cos_uv(dota_user[ucols].values, dota_item[icols].values) 
    eud_ui=eud_uv(dota_user[ucols].values, dota_item[icols].values)

    ui,ui_dot,ui_cos,ui_eud = [],[],[],[]

    for i,u in enumerate(dota_user['userId'].values):
        for j,ii in enumerate(dota_item['itemId'].values):
            ui.append(u+'_'+ii)
            ui_dot.append(dot_ui[i,j])
            ui_cos.append(cos_ui[i,j])
            ui_eud.append(eud_ui[i,j])
    return pd.DataFrame({'ui':ui,'score_{}'.format(flag):ui_dot,'cos_score_{}'.format(flag):ui_cos,'eud_score_{}'.format(flag):ui_eud})
    
def embs_ui_scores(df_user, df_item):
    dota_user = df_user.drop_duplicates('userId',keep='first')
    dota_item = df_item.drop_duplicates('itemId',keep='first')
    ui_w2v_tfidf = None

    for i,flag in enumerate(['dota']):
        print(i,flag)
        cross_ucols = [c for c in df_user.columns if 'w2v' in c and flag in c]
        cross_icols = [c for c in df_item.columns if 'w2v' in c and flag in c]
        if i == 0:
            ui_w2v_tfidf = dota_ui_score(dota_user = dota_user, dota_item = dota_item, ucols = cross_ucols,icols = cross_icols, flag='w2v_{}'.format(flag))
        else:
            ui_w2v_tmp = dota_ui_score(dota_user = dota_user, dota_item = dota_item,  ucols = cross_ucols,icols = cross_icols, flag='w2v_{}'.format(flag))
            ui_w2v_tfidf = ui_w2v_tfidf.merge(ui_w2v_tmp, on ='ui', how ='left') 
    
    ui_w2v_tfidf = reduce_mem_usage(ui_w2v_tfidf)
    
#     # 内存缘故注释 
#     for i,flag in enumerate(['tfidf']):
#         print(i,flag)
#         cross_ucols = [c for c in df_user.columns if flag in c]
#         cross_icols = [c for c in df_item.columns if flag in c]
        
#         ui_tfidf_tmp = dota_ui_score(dota_user = dota_user, dota_item = dota_item, ucols = cross_ucols, icols = cross_icols, flag='w2v_{}'.format(flag))
#         ui_w2v_tfidf = ui_w2v_tfidf.merge(ui_tfidf_tmp, on ='ui', how ='left') 
#     ui_w2v_tfidf = reduce_mem_usage(ui_w2v_tfidf)
    
    return ui_w2v_tfidf

##################### J
def get_Source(valid_qrel, valid_run):
   # 正样本
    Train = copy.deepcopy(valid_qrel)
    Train['label'] = 1
    del Train['rating']
    gc.collect()
    # 负样本
    df_negative = valid_run.merge(valid_qrel, how='left', on='userId')
    df_negative['itemIds'] = df_negative['itemIds'].apply(lambda x: x.split(','))
    df_negative.apply(lambda row: row['itemIds'].remove(row['itemId']), axis=1)
    df_negative = df_negative[['userId','itemIds']]
    # 直接调用explode()方法
    df_negative = df_negative.explode('itemIds').reset_index(drop=True)
    df_negative.columns = ['userId', 'itemId']
    df_negative['label'] = 0
    Train = pd.concat([Train, df_negative], ignore_index=True)
    return Train

def get_init_Test(test_run):
    test_run['itemId'] = test_run['itemIds'].map(lambda x:x.split(','))
    test_run = test_run[['userId', 'itemId']]
    test_run = test_run.explode('itemId').reset_index(drop=True)
    return test_run
    
def get_w2v_embedding(train,name,col1,col2,re_train,k=16):
    if re_train:
        customers = train[col1].unique().tolist()

        item_bought_by_people = train.groupby([col2])[col1].apply(lambda x: list(x)).reset_index()
        item_bought_by_people = item_bought_by_people[col1].values.tolist()

        from gensim.models.word2vec import Word2Vec 
        sentences = item_bought_by_people
        model = Word2Vec(sentences, size=k, sg=1, window=5, seed=2020, workers=24, min_count=1, iter=10)
        print(model)
        array = []
        for i in customers:
            array.append(model[i])
        dic = dict()
        for i in customers:  
            dic.update({i:model[i]})
        np.save(name,dic)
        user2Vec = pd.DataFrame(array)
        for i in range(k):
            user2Vec = user2Vec.rename(columns={user2Vec.columns[i]:(col1+'%d'%i)})
        user2Vec[col1] = customers
        return user2Vec
    else:
        customers = train[col1].unique().tolist()  
        word2vec_1 = np.load(name+'.npy',allow_pickle=True)
        array = []
        for i in tqdm(customers):
            array.append(word2vec_1.item()[i])
        user2Vec = pd.DataFrame(array)
        for i in range(k):
            user2Vec = user2Vec.rename(columns={user2Vec.columns[i]:(col1+'%d'%i)})
        user2Vec[col1] = customers
        return user2Vec
    
def tfidf_svd(data, f1,f2, n_components = 16): 
    tmp = data.groupby([f1])[f2].agg(list).reset_index() # : ' '.join(list((x).astype('str')))
    tmp.columns= [f1, '_list'] 
    tmp['_list'] = tmp['_list'].map(lambda x: ' '.join(x))  
    tfidf = TfidfVectorizer(max_df = 0.95, min_df = 2, sublinear_tf = True)
    res = tfidf.fit_transform(tmp['_list'])
    
    svd = TruncatedSVD(n_components = n_components, random_state = 2021)
    svd_res = svd.fit_transform(res)
    
    for i in range(n_components):
        tmp[f'{f1}_{f2}_tfidf_svd_{i}'] = svd_res[:,i]
        tmp[f'{f1}_{f2}_tfidf_svd_{i}'] = tmp[f'{f1}_{f2}_tfidf_svd_{i}'].astype(np.float32)
    del tmp['_list']
    return tmp
    
# 统计特征
def unique_num(x):
    return len(np.unique(x))

def get_Statistical_Features(train):
    ## 每个人购买多少次商品
    ## 每个人对商品的平均评分
    ## 每个人购买多少种商品

    ## 每个商品被购买多少次
    ## 每个商品的平均评分
    ## 每个商品被多少人购买
    user_buy_count = pd.DataFrame(train.groupby('userId').count()['itemId']).reset_index()
    user_buy_count = user_buy_count.rename(columns={user_buy_count.columns[1]:'user_buy_mean_count'})
    user_buy_rating = train.groupby('userId')['rating'].mean().reset_index()
    user_buy_rating = user_buy_rating.rename(columns={user_buy_rating.columns[1]:'user_buy_mean_rating'})
    user_buy_unique = train.groupby('userId').agg({'itemId': unique_num}).reset_index()
    user_buy_unique = user_buy_unique.rename(columns={user_buy_unique.columns[1]:'user_buy_unique'})

    item_buy_count = pd.DataFrame(train.groupby('itemId').count()['userId']).reset_index()
    item_buy_count = item_buy_count.rename(columns={item_buy_count.columns[1]:'item_buy_mean_count'})
    item_buy_traing = train.groupby('itemId')['rating'].mean().reset_index()
    item_buy_traing = item_buy_traing.rename(columns={item_buy_traing.columns[1]:'item_buy_mean_rating'})
    item_buy_unique = train.groupby('itemId').agg({'userId': unique_num}).reset_index()
    item_buy_unique = item_buy_unique.rename(columns={item_buy_unique.columns[1]:'item_buy_unique'})

    # 用户的各等级评分，以及评分比例
    userid = 'userId'
    df_cnt = pd.DataFrame()
    df_cnt[userid] = train[userid].unique().tolist() 

    temp = train
    df_cnt['rating_cnt_all'] = df_cnt[userid].map(temp[userid].value_counts())
    for i in range(6):
        temp = train[train['rating']==i]
        df_cnt['rating_cnt_'+str(i)] = df_cnt[userid].map(temp[userid].value_counts())
    df_cnt = df_cnt.fillna(0)

    for i in range(6):
        temp = train[train['rating']==i]
        df_cnt['rating_cnt_'+str(i)+'_rate'] = df_cnt['rating_cnt_'+str(i)]/(df_cnt['rating_cnt_all'])   
    del df_cnt['rating_cnt_all']

    # 商品的各等级评分，以及评分比例
    itemId = 'itemId'
    df_cnt1 = pd.DataFrame()
    df_cnt1[itemId] = train[itemId].unique().tolist() 

    temp = train
    df_cnt1['rating_cnt_all'] = df_cnt1[itemId].map(temp[itemId].value_counts())
    for i in range(6):
        temp = train[train['rating']==i]
        df_cnt1['rating_cnt_'+str(i)] = df_cnt1[itemId].map(temp[itemId].value_counts())
    df_cnt1 = df_cnt1.fillna(0)

    for i in range(6):
        temp = train[train['rating']==i]
        df_cnt1['rating_cnt_'+str(i)+'_rate'] = df_cnt1['rating_cnt_'+str(i)]/(df_cnt1['rating_cnt_all'])   
    del df_cnt1['rating_cnt_all']

    # 曾经购买次数
    user_item_buy_times = train.groupby(['userId','itemId'])['rating'].count().reset_index()
    user_item_buy_times = user_item_buy_times.rename({'rating':'user_item_buy_times'})

    user_item_buy_times_mean = train.groupby(['userId','itemId'])['rating'].mean().reset_index()
    user_item_buy_times_mean = user_item_buy_times_mean.rename({'rating':'user_item_buy_rating_mean'})
    

    return [user_buy_count, user_buy_rating, user_buy_unique, item_buy_count, item_buy_traing, item_buy_unique, df_cnt1, df_cnt, user_item_buy_times, user_item_buy_times_mean]

def get_Statistical_Features_added(train):
    ## 每个人购买多少次商品
    ## 每个人对商品的平均评分
    ## 每个人购买多少种商品

    ## 每个商品被购买多少次
    ## 每个商品的平均评分
    ## 每个商品被多少人购买
    name = 'added'
    user_buy_count = pd.DataFrame(train.groupby('userId').count()['itemId']).reset_index()
    user_buy_count = user_buy_count.rename(columns={user_buy_count.columns[1]:'user_buy_mean_count'+name})
    user_buy_rating = train.groupby('userId')['rating'].mean().reset_index()
    user_buy_rating = user_buy_rating.rename(columns={user_buy_rating.columns[1]:'user_buy_mean_rating'+name})
    user_buy_unique = train.groupby('userId').agg({'itemId': unique_num}).reset_index()
    user_buy_unique = user_buy_unique.rename(columns={user_buy_unique.columns[1]:'user_buy_unique'+name})

    item_buy_count = pd.DataFrame(train.groupby('itemId').count()['userId']).reset_index()
    item_buy_count = item_buy_count.rename(columns={item_buy_count.columns[1]:'item_buy_mean_count'+name})
    item_buy_traing = train.groupby('itemId')['rating'].mean().reset_index()
    item_buy_traing = item_buy_traing.rename(columns={item_buy_traing.columns[1]:'item_buy_mean_rating'+name})
    item_buy_unique = train.groupby('itemId').agg({'userId': unique_num}).reset_index()
    item_buy_unique = item_buy_unique.rename(columns={item_buy_unique.columns[1]:'item_buy_unique'+name})

    # 用户的各等级评分，以及评分比例
    userid = 'userId'
    df_cnt = pd.DataFrame()
    df_cnt[userid] = train[userid].unique().tolist() 

    temp = train
    df_cnt['rating_cnt_all'+name] = df_cnt[userid].map(temp[userid].value_counts())
    for i in range(6):
        temp = train[train['rating']==i]
        df_cnt['rating_cnt_'+str(i)+name] = df_cnt[userid].map(temp[userid].value_counts())
    df_cnt = df_cnt.fillna(0)

    for i in range(6):
        temp = train[train['rating']==i]
        df_cnt['rating_cnt_'+str(i)+'_rate'+name] = df_cnt['rating_cnt_'+str(i)+name]/(df_cnt['rating_cnt_all'+name])   
    del df_cnt['rating_cnt_all'+name]

    # 商品的各等级评分，以及评分比例
    itemId = 'itemId'
    df_cnt1 = pd.DataFrame()
    df_cnt1[itemId] = train[itemId].unique().tolist() 

    temp = train
    df_cnt1['rating_cnt_all'+name] = df_cnt1[itemId].map(temp[itemId].value_counts())
    for i in range(6):
        temp = train[train['rating']==i]
        df_cnt1['rating_cnt_'+str(i)+name] = df_cnt1[itemId].map(temp[itemId].value_counts())
    df_cnt1 = df_cnt1.fillna(0)

    for i in range(6):
        temp = train[train['rating']==i]
        df_cnt1['rating_cnt_'+str(i)+'_rate'+name] = df_cnt1['rating_cnt_'+str(i)+name]/(df_cnt1['rating_cnt_all'+name])   
    del df_cnt1['rating_cnt_all'+name]

    # 曾经购买次数
    user_item_buy_times = train.groupby(['userId','itemId'])['rating'].count().reset_index()
    user_item_buy_times = user_item_buy_times.rename({'rating':'user_item_buy_times'})

    user_item_buy_times_mean = train.groupby(['userId','itemId'])['rating'].mean().reset_index()
    user_item_buy_times_mean = user_item_buy_times_mean.rename({'rating':'user_item_buy_rating_mean'})

    return [user_buy_count, user_buy_rating, user_buy_unique, item_buy_count, item_buy_traing, item_buy_unique, df_cnt1, df_cnt, user_item_buy_times, user_item_buy_times_mean]

def get_Train(valid_qrel, valid_run, feature_list, embeddings):
   # 正样本
    Train = copy.deepcopy(valid_qrel)
    Train['label'] = 1
    del Train['rating']
    gc.collect()
    # 负样本
    df_negative = valid_run.merge(valid_qrel, how='left', on='userId')
    df_negative['itemIds'] = df_negative['itemIds'].apply(lambda x: x.split(','))
    df_negative.apply(lambda row: row['itemIds'].remove(row['itemId']), axis=1)
    df_negative = df_negative[['userId','itemIds']]
    # 直接调用explode()方法
    df_negative = df_negative.explode('itemIds').reset_index(drop=True)
    df_negative.columns = ['userId', 'itemId']
    df_negative['label'] = 0
    Train = pd.concat([Train, df_negative], ignore_index=True)
    # 拼接特征
    for ff in feature_list:
        for f in ff:
            Train = Train.merge(f, how='left')
    for embedding in embeddings:
        Train = Train.merge(embedding, how='left')
    Train = Train.fillna(0)
    return Train

def get_Test(test_run, feature_list, embeddings):
    test_run['itemId'] = test_run['itemIds'].map(lambda x:x.split(','))
    test_run = test_run[['userId', 'itemId']]
    test_run = test_run.explode('itemId').reset_index(drop=True)
    # 拼接特征
    for ff in feature_list:
        for f in ff:
            test_run = test_run.merge(f, how='left')
    for embedding in embeddings:
        test_run = test_run.merge(embedding, how='left')

    test_run = test_run.fillna(0)
    return test_run

# 三种计算相似性的方式
## 得到Jaccard公式相关性字典
def item_cf(df):  
    user_col = 'userId'
    item_col = 'itemId'
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()  
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))  
  
    sim_item = {}  
    item_cnt = defaultdict(int)  
    for user, items in user_item_dict.items():  
        for item in items:  
            item_cnt[item] += 1  
            sim_item.setdefault(item, {})  
            for relate_item in items:  
                if item == relate_item:  
                    continue 
              
                sim_item[item].setdefault(relate_item, 0)  
                sim_item[item][relate_item] += 1 / math.log(1 + len(items))
              
    sim_item_corr = sim_item.copy()  
    for i, related_items in tqdm(sim_item.items()):  
        for j, cij in related_items.items():  
            sim_item_corr[i][j] = cij / math.sqrt(item_cnt[i]*item_cnt[j]) 

    return sim_item_corr

## Jaccard相似度矩阵
def cal_jaccard(train,embedding_items):
    embedding = copy.deepcopy(embedding_items)
    l = embedding['itemId']
    array = []

    item_sim_list=item_cf(train)

    for item in l:
        temp = []
        for item1 in l:
            if item1 in item_sim_list[item].keys():
                temp.append(item_sim_list[item][item1])
            else:
                temp.append(0)
        array.append(temp)
    return array

def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

## cos相似度
def cal_cos(embedding_items):
    embedding = copy.deepcopy(embedding_items)
    l = embedding['itemId']
    del embedding['itemId']
    array = np.array(embedding)
    column_lst = l
    array = get_cos_similar_matrix(array,array)
    return array

from scipy.spatial.distance import pdist, squareform

## l1距离
def cal_l1(embedding_items):
    embedding = copy.deepcopy(embedding_items)
    l = embedding['itemId']
    del embedding['itemId']
    array = np.array(embedding)
    column_lst = l
    array = pdist(array, 'cityblock')
    array = squareform(array)
    return array

## l2距离
def cal_l2(embedding_items):
    embedding = copy.deepcopy(embedding_items)
    l = embedding['itemId']
    del embedding['itemId']
    array = np.array(embedding)
    column_lst = l
    array = pdist(array, 'euclidean')
    array = squareform(array)
    return array

## 皮尔逊相关系数
def cal_person(embedding_items):
    embedding = copy.deepcopy(embedding_items)
    l = embedding['itemId']
    del embedding['itemId']
    array = np.array(embedding)
    unstrtf_lst = array
    column_lst = l

    # 计算列表两两间的皮尔逊相关系数
    data_dict = {} # 创建数据字典，为生成Dataframe做准备
    for col, gf_lst in zip(column_lst, unstrtf_lst):
        data_dict[col] = gf_lst
    unstrtf_df = pd.DataFrame(data_dict)
    cor1 = unstrtf_df.corr() # 计算相关系数，得到一个矩阵
  
    sim_item_corr = cor1.to_numpy()
    return sim_item_corr

# 计算当前商品和当前用户曾购买商品的相似度
def get_bought(train,data):
    valid_item_ = data.groupby('userId')['itemId'].agg(list).reset_index()  
    train_item = train.groupby('userId')['itemId'].agg(list).reset_index() 
    user_item_dict = dict(zip(train_item['userId'], train_item['itemId']))

    # 获得曾经买过的东西list
    df = pd.DataFrame()
    df['userId'] = valid_item_['userId']
    bought = []
    for user in list(df['userId'].values):
        bought.append(user_item_dict[user])
    df['bought_items'] = bought
    
    data = pd.merge(data,df,how='left')
    return data

## 计算相似度
def cal_sim(train,data,dic,sim,name):
    data = get_bought(train,data) 
    train_items = set(train['itemId'].values)
    temp_item = data['itemId'].values
    bought_items = data['bought_items'].values

    j = 0
    for temp_sim in sim:
        print("*********** cal "+name[j]+" sim***********")
        rel = []
        i = 0
        for t in temp_item:
            i1 = temp_item[i]
            li = []
            if i1 in train_items:
                i1 = dic[i1]
                for i2 in bought_items[i]:
                    if i2 in train_items:
                        i2 = dic[i2]
                        li.append(temp_sim[i1][i2])
            rel.append(li)
            i+=1

        # 取出top5
        array = []
        for re in rel:
            re = np.sort(re)
            features = []
            for i in range(1,6):
                try:
                  # 什么意思
                    max_val = re[-1 * i]
                except:
                    max_val = 0
                features.append(max_val)
            array.append(features)

        array = np.array(array)


        # 统计
        item_sum = np.sum(array,axis=1)
        item_max = np.max(array,axis=1)
        item_mean = np.mean(array,axis=1)
        item_min = np.min(array,axis=1)
        item_median = np.median(array,axis=1)
        item_std = np.std(array,axis=1)

        data['item_sum_'+name[j]] = item_sum
        data['item_max_'+name[j]] = item_max
        data['item_mean_'+name[j]] = item_mean
        data['item_min_'+name[j]] = item_min
        data['item_median_'+name[j]] = item_median
        data['item_std_'+name[j]] = item_std
        j+=1
    
    del data['bought_items']
    return data

# 单个数据,和其embedding,所有源的数据
def agg_sim(train,embedding):
    train_sim1_t1 = cal_jaccard(train,embedding)
    train_sim2_t1 = cal_cos(embedding)
    return [train_sim1_t1,train_sim2_t1]

def cal_sim_for_all(train1,train2,Tric_train1,Tric_train2,\
          Train1,Train2,Test1,Test2,\
          embedding_t1_items,embedding_t2_items,embedding_t1_items_test,embedding_t2_items_test,\
          feature_name):
    s1 = embedding_t1_items['itemId']
    s2 = embedding_t2_items['itemId']
    s3 = embedding_t1_items_test['itemId']
    s4 = embedding_t2_items_test['itemId']
    l1 = range(len(s1))
    l2 = range(len(s2))
    l3 = range(len(s3))
    l4 = range(len(s4))

    dic1 = dict(zip(s1,l1))
    dic2 = dict(zip(s2,l2))
    dic3 = dict(zip(s3,l3))
    dic4 = dict(zip(s4,l4))

    # train_all计算
    train_all = [train1,train2,Tric_train1,Tric_train2]
    target_all = [Train1,Train2,Test1,Test2]
    embedding_all = [embedding_t1_items,embedding_t2_items,embedding_t1_items_test,embedding_t2_items_test]
    name = ['valid_t1','valid_t2','test_t1','test_t2']
    dic_all = [dic1,dic2,dic3,dic4]
  
    tt = []
    for i in range(4):
        print('****************'+name[i]+'*****************')
        train = train_all[i]
        embedding = embedding_all[i]
        target = target_all[i]
        train_sim = agg_sim(train,embedding)
        dic = dic_all[i]
        target = cal_sim(train,target,dic,train_sim,feature_name)
        tt.append(target)
    return tt

def get_kfold_users(trn_df, n=5, seed=2022):
    import random
    random.seed(seed)
    user_ids = trn_df['userId'].unique()
    random.shuffle(user_ids)
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set

# 排序结果归一化
def norm_sim(sim_df, weight=0.0):
    # print(sim_df.head())
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))

    sim_df = sim_df.apply(lambda sim: sim + weight)  # plus one
    return sim_df


def stack_model(oof_1, oof_2, oof_3, predictions_1, predictions_2, predictions_3, y):
    train_stack = pd.concat([oof_1, oof_2, oof_3], axis=1)
    test_stack = pd.concat([predictions_1, predictions_2, predictions_3], axis=1)
    
    oof = np.zeros((train_stack.shape[0],))
    predictions = np.zeros((test_stack.shape[0],))
    scores = []
    
    from sklearn.model_selection import RepeatedKFold
    folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2021)
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, train_stack)): 
        print("fold n°{}".format(fold_+1))
        trn_data, trn_y = train_stack.loc[trn_idx], y[trn_idx]
        val_data, val_y = train_stack.loc[val_idx], y[val_idx]
        
        clf = Ridge(random_state=2021)
        clf.fit(trn_data, trn_y)

        oof[val_idx] = clf.predict(val_data)
        predictions += clf.predict(test_stack) / (5 * 2)
        
        score_single = roc_auc_score(val_y, oof[val_idx])
        scores.append(score_single)
        print(f'{fold_+1}/{5}', score_single)
    print('mean: ',np.mean(scores))
   
    return oof, predictions