import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
import math
import gensim

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

from tools import *

## s1 s2 s3
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

## t1 t2
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

def item_cf(df, user_col, item_col):  
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()  
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))  
    
    user_rate_ = df.groupby(user_col)['rating'].agg(list).reset_index() # 引入rating因素
    user_rate_dict = dict(zip(user_rate_[user_col], user_rate_['rating']))
    
    sim_item = {}  
    item_cnt = defaultdict(int)  
    for user, items in tqdm(user_item_dict.items()):  
        for loc1, item in enumerate(items):  
            item_cnt[item] += 1  
            sim_item.setdefault(item, {})  
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:  
                    continue 
                r1 = user_rate_dict[user][loc1] # rating提取
                r2 = user_rate_dict[user][loc2]
                sim_item[item].setdefault(relate_item, 0)  
                sim_item[item][relate_item] += 1 / math.log(1 + len(items))
                
    sim_item_corr = sim_item.copy()  
    for i, related_items in tqdm(sim_item.items()):  
        for j, cij in related_items.items():  
            sim_item_corr[i][j] = cij / math.sqrt(item_cnt[i]*item_cnt[j]) 
  
    return sim_item_corr, user_item_dict  

def user_cf(df, user_col, item_col):
    
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()  
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col])) 
    
    item_user_ = df.groupby(item_col)[user_col].agg(list).reset_index()  
    item_user_dict = dict(zip(item_user_[item_col], item_user_[user_col])) 
    
    user_time_dict = {}
    for items in df[[user_col,item_col,'rating']].values:
        i1, i2, i3 = items[0], items[1], items[2]
        try:
            user_time_dict[i1][i2] = i3
        except:
            user_time_dict[i1] = {}
            user_time_dict[i1][i2] = i3

    sim_user = {}
    user_cnt = defaultdict(int)
    for item, users in tqdm(item_user_dict.items()):
        num_users = len(users)
        for loc1, u in enumerate(users):
            user_cnt[u] += 1
            sim_user.setdefault(u, {})
            for loc2, relate_user in enumerate(users):
                if u == relate_user:
                    continue
                t1 = user_time_dict[u][item] # rating提取
                t2 = user_time_dict[relate_user][item]
                sim_user[u].setdefault(relate_user, 0)
                weight = 1.0
                sim_user[u][relate_user] += weight / math.log(1 + num_users) 

    sim_user_corr = sim_user.copy()
    for u, related_users in tqdm(sim_user.items()):
        for v, cuv in related_users.items():
            sim_user_corr[u][v] = cuv / math.sqrt(user_cnt[u] * user_cnt[v])

    return sim_user_corr, user_item_dict

def swing_base(df, user_col, item_col):
    
    u_items = dict()
    i_users = dict()
    for index, row in tqdm(df.iterrows()):
        u_items.setdefault(row[user_col], set())
        i_users.setdefault(row[item_col], set())

        u_items[row[user_col]].add(row[item_col])
        i_users[row[item_col]].add(row[user_col])
    
    item_pairs = list(combinations(i_users.keys(), 2))
    item_sim_dict = dict()
    cnt = 0
    alpha = 5.0
    for (i, j) in tqdm(item_pairs):
        cnt += 1
        user_pairs = list(combinations(i_users[i] & i_users[j], 2))
        result = 0.0
        for (u, v) in user_pairs:
            result += 1 / (alpha + list(u_items[u] & u_items[v]).__len__())

        item_sim_dict.setdefault(i, dict())
        item_sim_dict[i][j] = result
    
    sim_item_corr = dict()
    for item, sim_items in tqdm(item_sim_dict.items()):
        sim_item_corr.setdefault(item, dict())
        sim_item_corr[item] = dict(sorted(sim_items.items(), key = lambda k:k[1], reverse=True))
    
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()  
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col])) 
    
    return sim_item_corr, user_item_dict

def make_user_time_tuple(group_df, user_col='user_id', item_col='item_id', time_col='time'):
    user_time_tuples = list(zip(group_df[user_col], group_df[time_col]))
    return user_time_tuples

def swing_new(df, user_col='user_id', item_col='item_id', time_col='time'):
    # 1. item, (u1,t1), (u2, t2).....
    df['time'] = 1
    item_user_df = df.sort_values(by=[item_col, time_col])
    item_user_df = item_user_df.groupby(item_col).apply(
        lambda group: make_user_time_tuple(group, user_col, item_col, time_col)).reset_index().rename(
        columns={0: 'user_id_time_list'})
    item_user_time_dict = dict(zip(item_user_df[item_col], item_user_df['user_id_time_list']))

    user_item_time_dict = defaultdict(list)
    # 2. ((u1, u2), i1, d12)
    u_u_cnt = defaultdict(list)
    item_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, u_time in user_time_list:
            # just record
            item_cnt[item] += 1
            user_item_time_dict[u].append((item, u_time))

            for relate_u, relate_u_time in user_time_list:
                if relate_u == u:
                    continue

                key = (u, relate_u) if u <= relate_u else (relate_u, u)
                u_u_cnt[key].append((item, np.abs(u_time - relate_u_time)))

    # 3. (i1,i2), sim
    sim_item = {}
    alpha = 5.0
    for u_u, co_item_times in tqdm(u_u_cnt.items()):
        num_co_items = len(co_item_times)
        for i, i_time_diff in co_item_times:
            sim_item.setdefault(i, {})
            for j, j_time_diff in co_item_times:
                if j == i:
                    continue
                weight = 1.0 
                sim_item[i][j] = sim_item[i].setdefault(j, 0.) + weight / (alpha + num_co_items)
    
    # 4. norm by item count
    sim_item_corr = sim_item.copy()
    for i, related_items in tqdm(sim_item.items()):
        for j, cij in related_items.items():
            sim_item_corr[i][j] = cij / math.sqrt(item_cnt[i] * item_cnt[j])
    
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()  
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))
    
    return sim_item_corr, user_item_dict

def bi_graph(df, user_col, item_col):
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()  
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col])) 
    
    item_user_ = df.groupby(item_col)[user_col].agg(list).reset_index()  
    item_user_dict = dict(zip(item_user_[item_col], item_user_[user_col])) 
    
    item_cnt = defaultdict(int)
    for user, items in tqdm(user_item_dict.items()):
        for i in items:
            item_cnt[i] += 1

    sim_item = {}

    for item, user_lists in tqdm(item_user_dict.items()):

        sim_item.setdefault(item, {})
        
        for u in user_lists:
            try:
                tmp_len = len(item_user_dict[u])
            except:
                tmp_len = 0.1
            for relate_item in user_item_dict[u]:
                sim_item[item].setdefault(relate_item, 0)
                weight = 1
                sim_item[item][relate_item] += weight / (math.log(len(user_lists) + 1) * math.log(tmp_len + 1))

    return sim_item, user_item_dict

def personal_rank(df, user_col, item_col, roots):
    
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index() 
    item_user_ = df.groupby(item_col)[user_col].agg(list).reset_index()

    graph_dict = dict(zip(user_item_[user_col], user_item_[item_col]))
    tmp_dict = dict(zip(item_user_[item_col], item_user_[user_col]))
    for itemId, itemId_list in tmp_dict.items():
        graph_dict[itemId] = itemId_list
    
    final_sim = {}
    for root in tqdm(roots):
        rank={}
        rank={point:0 for point in graph_dict}  #将所有顶点的PR值初始为0
        rank[root]=1  #固定顶点的PR值为1
        alpha = 0.85
        recom_result={}
        for item_index in range(10):
            tmp_rank={}    #临时存放结果
            tmp_rank={point:0 for point in graph_dict}
            for out_point,out_dict in graph_dict.items(): #对于graph中每一个字典对
                for inner_point in graph_dict[out_point]:
                    #每一个节点的PR值等于所有有边指向当前节点的节点的PR值的等分
                    tmp_rank[inner_point]+=round(alpha*rank[out_point]/len(out_dict),4)
                    if inner_point==root:
                        tmp_rank[inner_point]+=round(1-alpha,4)
            if tmp_rank==rank:
                break    #提前结束迭代
            rank=tmp_rank
            
        final_sim[root] = rank
    
    return final_sim, graph_dict

def recommend(sim_item_corr, user_item_dict, user_id):  
    rank = {}
    try:
        interacted_items = user_item_dict[user_id]
    except:
        interacted_items = {}
    for i in interacted_items: 
        try:
            for j, wij in sorted(sim_item_corr[i].items(), key=lambda d: d[1], reverse=True):  
                if j not in interacted_items:
                    rank.setdefault(j, 0) 
                    rank[j] += wij
        except:
            pass

    return sorted(rank.items(), key=lambda d: d[1], reverse=True)

def recommend_ucf(sim_user_corr, user_item_dict, user_id):  
    rank = {}
    try:
        interacted_items = user_item_dict[user_id]
    except:
        interacted_items = {}
    try:
        for relate_user, score_similar in sorted(sim_user_corr[user_id].items(), key=lambda d: d[1], reverse=True):
            for i, item in enumerate(user_item_dict[relate_user]):
                if item not in interacted_items:
                    rank.setdefault(item, 0) 
                    rank[item] += score_similar
    except:
        pass
    
    return sorted(rank.items(), key=lambda d: d[1], reverse=True)

def match_func(items1, items2):
    res = []
    for it in items1:
        if it in items2:
            res.append(it)
    if len(res) < 100:
        for it in items2:
            if it not in res:
                res.append(it)
    return res[:100]

def recall_feat_func(train, valid_run, flag):
    # 构建相似矩阵
    print('item_cf...')
    item_sim_list1, user_item = item_cf(train, 'userId', 'itemId')
    print('user_cf...')
    item_sim_list2, user_item = user_cf(train, 'userId', 'itemId')
        
    # item_cf
    recom_item = []
    for i in tqdm(valid_run['userId'].unique()):  
        rank_item = recommend(item_sim_list1, user_item, i)  
        for j in rank_item:  
            recom_item.append([i, j[0], j[1]])  
    recom_item_df1 = pd.DataFrame(recom_item)
    recom_item_df1.columns = ['userId','itemId','score1_'+flag]
    
    # user_cf
    recom_item = []
    for i in tqdm(valid_run['userId'].unique()):  
        rank_item = recommend_ucf(item_sim_list2, user_item, i)  
        for j in rank_item:  
            recom_item.append([i, j[0], j[1]])  
    recom_item_df2 = pd.DataFrame(recom_item)
    recom_item_df2.columns = ['userId','itemId','score2_'+flag]
    
    recom_item_df1 = recom_item_df1.merge(recom_item_df2, on=['userId','itemId'], how='left')
    
    return recom_item_df1

def recall_feat_swing_func(train, valid_run, flag):
    # 构建相似矩阵
    print('item_cf...')
    item_sim_list1, user_item = item_cf(train, 'userId', 'itemId')
    print('swing_new...')
    item_sim_list2, user_item = swing_new(train, 'userId', 'itemId')
    print('swing_base...')
    item_sim_list3, user_item = swing_base(train, 'userId', 'itemId')
    print('user_cf...')
    item_sim_list4, user_item = user_cf(train, 'userId', 'itemId')
    print('bi_graph...')
    item_sim_list5, user_item = bi_graph(train, 'userId', 'itemId')
        
    # item_cf
    recom_item = []
    for i in tqdm(valid_run['userId'].unique()):  
        rank_item = recommend(item_sim_list1, user_item, i)  
        for j in rank_item:  
            recom_item.append([i, j[0], j[1]])  
    recom_item_df1 = pd.DataFrame(recom_item)
    recom_item_df1.columns = ['userId','itemId','score1_'+flag]
    
    # swing_new
    recom_item = []
    for i in tqdm(valid_run['userId'].unique()):  
        rank_item = recommend(item_sim_list2, user_item, i)  
        for j in rank_item:  
            recom_item.append([i, j[0], j[1]])  
    recom_item_df2 = pd.DataFrame(recom_item)
    recom_item_df2.columns = ['userId','itemId','score2_'+flag]
    
    # swing_base
    recom_item = []
    for i in tqdm(valid_run['userId'].unique()):  
        rank_item = recommend(item_sim_list3, user_item, i)  
        for j in rank_item:  
            recom_item.append([i, j[0], j[1]])  
    recom_item_df3 = pd.DataFrame(recom_item)
    recom_item_df3.columns = ['userId','itemId','score3_'+flag]
    
    # user_cf
    recom_item = []
    for i in tqdm(valid_run['userId'].unique()):  
        rank_item = recommend_ucf(item_sim_list4, user_item, i)  
        for j in rank_item:  
            recom_item.append([i, j[0], j[1]])  
    recom_item_df4 = pd.DataFrame(recom_item)
    recom_item_df4.columns = ['userId','itemId','score4_'+flag]
    
    # bi_graph
    recom_item = []
    for i in tqdm(valid_run['userId'].unique()):  
        rank_item = recommend(item_sim_list5, user_item, i)  
        for j in rank_item:  
            recom_item.append([i, j[0], j[1]])  
    recom_item_df5 = pd.DataFrame(recom_item)
    recom_item_df5.columns = ['userId','itemId','score5_'+flag]
    
    recom_item_df1 = recom_item_df1.merge(recom_item_df2, on=['userId','itemId'], how='left')
    recom_item_df1 = recom_item_df1.merge(recom_item_df3, on=['userId','itemId'], how='left')
    recom_item_df1 = recom_item_df1.merge(recom_item_df4, on=['userId','itemId'], how='left')
    recom_item_df1 = recom_item_df1.merge(recom_item_df5, on=['userId','itemId'], how='left')
    
    return recom_item_df1

s1 = pd.concat([train_5core_s1,train_s1,valid_qrel_s1], axis=0)
s2 = pd.concat([train_5core_s2,train_s2,valid_qrel_s2], axis=0)
s3 = pd.concat([train_5core_s3,train_s3,valid_qrel_s3], axis=0)

t1 = pd.concat([train_5core_t1,train_t1], axis=0)
t2 = pd.concat([train_5core_t2,train_t2], axis=0)

def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)
mkdir('./recall_feature_data/')

for i, items in enumerate([
              [[t1],[t2]],
              [[t1,t2],[t1,t2]],
              [[s1,s2,s3,t1],[s1,s2,s3,t2]],
              [[s1,s2,s3,t1,t2],[s1,s2,s3,t1,t2]],
             ]):
    item1, item2 = items[0], items[1]
    print('====================================== {} ======================================'.format(str(i)))
    
    if i <= 1:
        recall_func = recall_feat_swing_func
    else:
        recall_func = recall_feat_func
    
    recom_valid_t1_df = recall_func(pd.concat(item1, axis=0), valid_run_t1, 'swing_'+str(i))
    recom_test_t1_df  = recall_func(pd.concat(item1+[valid_qrel_t1], axis=0), test_run_t1, 'swing_'+str(i))

    recom_valid_t2_df = recall_func(pd.concat(item2, axis=0), valid_run_t2, 'swing_'+str(i))
    recom_test_t2_df  = recall_func(pd.concat(item2+[valid_qrel_t2], axis=0), test_run_t2, 'swing_'+str(i))
    
    recom_valid_t1_df.to_csv('./recall_feature_data/recom_valid_t1_swing_feat{}.csv'.format(str(i)), index=False)
    recom_test_t1_df.to_csv('./recall_feature_data/recom_test_t1_swing_feat{}.csv'.format(str(i)), index=False)

    recom_valid_t2_df.to_csv('./recall_feature_data/recom_valid_t2_swing_feat{}.csv'.format(str(i)), index=False)
    recom_test_t2_df.to_csv('./recall_feature_data/recom_test_t2_swing_feat{}.csv'.format(str(i)), index=False)
    
    recall_func = []
