import torch
import random
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.sparse as sp
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

random.seed(0)


class Central_ID_Bank(object):
    """
    Central for all cross-market user and items original id and their corrosponding index values
    """
    def __init__(self):
        self.user_id_index = {}
        self.item_id_index = {}
        self.last_user_index = 0
        self.last_item_index = 0
        
    def query_user_index(self, user_id):
        if user_id not in self.user_id_index:
            self.user_id_index[user_id] = self.last_user_index
            self.last_user_index += 1
        return self.user_id_index[user_id]
    
    def query_item_index(self, item_id):
        if item_id not in self.item_id_index:
            self.item_id_index[item_id] = self.last_item_index
            self.last_item_index += 1
        return self.item_id_index[item_id]
    
    def query_user_id(self, user_index):
        user_index_id = {v:k for k, v in self.user_id_index.items()}
        if user_index in user_index_id:
            return user_index_id[user_index]
        else:
            print(f'USER index {user_index} is not valid!')
            return 'xxxxx'
        
    def query_item_id(self, item_index):
        item_index_id = {v:k for k, v in self.item_id_index.items()}
        if item_index in item_index_id:
            return item_index_id[item_index]
        else:
            print(f'ITEM index {item_index} is not valid!')
            return 'yyyyy'

    
    

class MetaMarket_DataLoader(object):
    """Data Loader for a few markets, samples task and returns the dataloader for that market"""
    
    def __init__(self, task_list, sample_batch_size, task_batch_size=2, shuffle=True, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        
        self.num_tasks = len(task_list)
        self.task_list = task_list
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.sample_batch_size = sample_batch_size
        self.task_list_loaders = {
            idx:DataLoader(task_list[idx], batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory) \
            for idx in range(len(self.task_list))
        }
        self.task_list_iters = {
            idx:iter(self.task_list_loaders[idx]) \
            for idx in range(len(self.task_list))
        }
        self.task_batch_size = min(task_batch_size, self.num_tasks)
    
    def refresh_dataloaders(self):
        self.task_list_loaders = {
            idx:DataLoader(self.task_list[idx], batch_size=self.sample_batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=False) \
            for idx in range(len(self.task_list))
        }
        self.task_list_iters = {
            idx:iter(self.task_list_loaders[idx]) \
            for idx in range(len(self.task_list))
        }
        
    def get_iterator(self, index):
        return self.task_list_iters[index]
        
    def sample_task(self):
        sampled_task_idx = random.randint(0, self.num_tasks-1)
#         print(f'task number {sampled_task_idx} sampled')
        return self.task_list_loaders[sampled_task_idx]
    
    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, index):
        return self.task_list_loaders[index]
    

        
        
class MetaMarket_Dataset(object):
    """
    Wrapper around market data (task)
    ratings: {
      0: us_market_gen,
      1: de_market_gen,
      ...
    }
    """
    def __init__(self, task_gen_dict, num_negatives=4, meta_split='train'):
        self.num_tasks = len(task_gen_dict)
        if meta_split=='train':
            self.task_gen_dict = {idx:cur_task.instance_a_market_train_task(idx, num_negatives) for idx, cur_task  in task_gen_dict.items()}
        else:
            self.task_gen_dict = {idx:cur_task.instance_a_market_valid_task(idx, split=meta_split) for idx, cur_task  in task_gen_dict.items()}
        
    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        return self.task_gen_dict[index]
        


class MarketTask(Dataset):
    """
    Individual Market data that is going to be wrapped into a metadataset  i.e. MetaMarketDataset

    Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset
    """
    def __init__(self, task_index, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.task_index = task_index
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

        
    def __len__(self):
        return self.user_tensor.size(0)
    
    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

def w2v_embedding(df, mk, gpk, vsize=24):
    df_mk_gpk_list = df.groupby([mk]).apply(lambda x: x[gpk].tolist()).reset_index()
    df_mk_gpk_list.columns = [mk, gpk+'_list']

    sentence = [[str(x) for x in x] for x in df_mk_gpk_list[gpk+'_list'].values.tolist()]
    model = Word2Vec(sentence, size=vsize, sg=1, window=10, seed=2009, workers=36, min_count=1)

    emb_matrix = []
    id = []
    for i,k in enumerate(tqdm(df_mk_gpk_list[gpk+'_list'].values)):
        tmp = np.zeros(shape=(vsize))
        for seq in k:
            tmp = tmp + model[str(seq)] / len(k)
        emb_matrix.append(tmp)
        id.append(df_mk_gpk_list[mk][i])

    return emb_matrix


class TaskGenerator(object):
    """Construct dataset"""

    def __init__(self, train_data, id_index_bank, valid_train, valid_pos, args):
        """
        args:
            train_data: pd.DataFrame, which contains 3 columns = ['userId', 'itemId', 'rating']
            id_index_bank: converts ids to indices 
        """

        self.id_index_bank = id_index_bank

        # None for evaluation purposes
        if train_data is not None: 
            self.ratings = train_data
            self.valid_pos = valid_pos
            self.valid_train = valid_train

            # replace ids with corrosponding index for both users and items
            self.ratings['userId'] = self.ratings['userId'].apply(lambda x: self.id_index_bank.query_user_index(x) )
            self.ratings['itemId'] = self.ratings['itemId'].apply(lambda x: self.id_index_bank.query_item_index(x) )

            self.valid_pos['userId'] = self.valid_pos['userId'].apply(lambda x: self.id_index_bank.query_user_index(x))
            self.valid_pos['itemId'] = self.valid_pos['itemId'].apply(lambda x: self.id_index_bank.query_item_index(x))

            self.valid_train['userId'] = self.valid_train['userId'].apply(lambda x: self.id_index_bank.query_user_index(x))
            self.valid_train['itemId'] = self.valid_train['itemId'].apply(lambda x: self.id_index_bank.query_item_index(x))

            # get item and user pools (indexed version)
            self.user_pool = set(self.ratings['userId'].unique()).union(set(self.valid_pos['userId'].unique())).union(set(self.valid_train['userId'].unique()))
            self.item_pool = set(self.ratings['itemId'].unique()).union(set(self.valid_pos['itemId'].unique())).union(set(self.valid_train['itemId'].unique()))

            self.train_ratings = self.ratings
            valid_ratings = pd.concat([self.valid_train,self.valid_pos],axis=0).drop_duplicates(subset=['userId', 'itemId'],keep='last')
            valid_ratings1 = pd.concat([self.valid_train, self.valid_pos], axis=0).drop_duplicates(subset=['userId', 'itemId'], keep=False)
            self.pos = valid_ratings1.append(valid_ratings).drop_duplicates(subset=['userId', 'itemId'], keep=False)
            self.valid_train_pos = pd.concat([self.valid_train,self.pos],axis=0).drop_duplicates(subset=['userId', 'itemId'], keep='last')
            # create negative item samples
            self.negatives_train = self._sample_negative(self.ratings)

            if 'with' in args.model_name:
                self.v_size = args.latent_dim
                self.user_matrix = w2v_embedding(self.ratings,'userId','itemId',self.v_size)
                self.item_matrix = w2v_embedding(self.ratings,'itemId','userId',self.v_size)

    
    def _sample_negative(self, ratings):
        a = pd.concat([ratings,self.valid_pos],axis=0)
        by_userid_group = a.groupby("userId")['itemId']
        negatives_train = {}
        for userid, group_frame in by_userid_group:
            pos_itemids = set(group_frame.values.tolist())
            neg_itemids = self.item_pool - pos_itemids
            neg_itemids_train = neg_itemids
            negatives_train[userid] = neg_itemids_train
        return negatives_train
                    
        
    def instance_a_market_train_task(self, index, num_negatives):
        """instance train task's torch Dataset"""
        users, items, ratings = [], [], []
        train_ratings = self.train_ratings
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            if float(row.rating)<=4.0:
                ratings.append((float(row.rating)/10+0.5)*0.95+0.05/2)
            else:
                ratings.append(1.0)


            cur_negs = self.negatives_train[int(row.userId)]
            cur_negs = random.sample(cur_negs, min(num_negatives, len(cur_negs)))
            for neg in cur_negs:
                users.append(int(row.userId))
                items.append(int(neg))
                ratings.append(float(0.05/2))  # negative samples get 0 rating

        for row in self.valid_train_pos.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            if float(row.rating) == 0.0:
                ratings.append(0.05/2)
            else:
                ratings.append(1.0)

        dataset = MarketTask(index, user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return dataset

    def normalized_adj_single(self, adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()

    def instance_a_graph(self, user_num, item_num):
        self.graph = sp.dok_matrix((user_num+item_num, user_num+item_num), dtype=np.float32)
        train_ratings = self.train_ratings
        for row in train_ratings.itertuples():
            if float(row.rating) <= 4.0:
                self.graph[int(row.userId), user_num+int(row.itemId)] = (float(row.rating)/10+0.5)*0.95+0.05/2
                self.graph[user_num+int(row.itemId), int(row.userId)] = (float(row.rating)/10+0.5)*0.95+0.05/2
            else:
                self.graph[int(row.userId), user_num + int(row.itemId)] = 1.
                self.graph[user_num + int(row.itemId), int(row.userId)] = 1.

        for row in self.pos.itertuples():
            self.graph[int(row.userId), user_num+int(row.itemId)] = 1.
            self.graph[user_num+int(row.itemId), int(row.userId)] = 1.

        return self.normalized_adj_single(self.graph).tocsr()
    
    def instance_a_market_train_dataloader(self, index, num_negatives, sample_batch_size, shuffle=True, num_workers=0):
        """instance train task's torch Dataloader"""
        dataset = self.instance_a_market_train_task(index, num_negatives)
        return DataLoader(dataset, batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        
    
    def load_market_test_run(self, valid_run_file):
        users, items, ratings = [], [], []
        with open(valid_run_file, 'r') as f:
            for line in f:
                linetoks = line.split('\t')
                user_id = linetoks[0]
                item_ids = linetoks[1].strip().split(',')
                for cindex, item_id in enumerate(item_ids):
                    users.append(self.id_index_bank.query_user_index(user_id))
                    items.append(self.id_index_bank.query_item_index(item_id))
                    ratings.append(float(0))


        dataset = MarketTask(0, user_tensor=torch.LongTensor(users),
                                            item_tensor=torch.LongTensor(items),
                                            target_tensor=torch.FloatTensor(ratings))
        return dataset

    def instance_a_market_test_dataloader(self, valid_run_file, sample_batch_size, shuffle=False, num_workers=0):
        """instance target market's validation data torch Dataloader"""
        dataset = self.load_market_test_run(valid_run_file)
        return DataLoader(dataset, batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers)

    def instance_a_market_valid_dataloader(self, X_valid, sample_batch_size, shuffle=False, num_workers=0):
        """instance target market's validation data torch Dataloader"""
        users, items, ratings = [], [], []

        for line in X_valid.itertuples():
            user_id = line.userId
            item_ids = line.itemIds.strip().split(',')
            for cindex, item_id in enumerate(item_ids):
                users.append(self.id_index_bank.query_user_index(user_id))
                items.append(self.id_index_bank.query_item_index(item_id))
                ratings.append(float(0))

        dataset = MarketTask(0, user_tensor=torch.LongTensor(users),
                                            item_tensor=torch.LongTensor(items),
                                            target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers)

    
