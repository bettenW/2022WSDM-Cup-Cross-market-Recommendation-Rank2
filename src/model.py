import torch
import pickle
import os
from utils import *
from validate_submission import *
import numpy as np
import torch.nn.functional as F
from data import *

class Model(object):
    def __init__(self, args, my_id_bank, valid_dataloader, graph, user_matrix=None, item_matrix=None):
        self.args = args
        self.my_id_bank = my_id_bank
        self.model_name = args.model_name
        self.model = self.prepare_model(graph,user_matrix,item_matrix)
        self.valid_dataloader = valid_dataloader

    
    def prepare_model(self, graph=None, user_matrix=None, item_matrix=None):
        if self.my_id_bank is None:
            print('ERR: Please load an id_bank before model preparation!')
            return None
            
        self.config = {'alias': 'model',
              'batch_size': self.args.batch_size, #1024,
              'optimizer': 'adam',
              'adam_lr': self.args.lr, #0.005, #1e-3,
              'latent_dim': self.args.latent_dim, #8
              'num_negative': self.args.num_negative, #4
              'l2_regularization': self.args.l2_reg, #1e-07,
              'use_cuda': torch.cuda.is_available() and self.args.cuda, #False,
              'device_id': 0,
              'embedding_user': None,
              'embedding_item': None,
              'save_trained': True,
              'num_users': int(self.my_id_bank.last_user_index+1), 
              'num_items': int(self.my_id_bank.last_item_index+1),
        }
        if 'no' in self.model_name:
            print('Model is LightGCN without word2vec!')
            self.model = LightGCN_no_w2v(self.config, graph)
        else:
            print('Model is LightGCN with word2vec!')
            self.model = LightGCN_with_w2v(self.config,graph,user_matrix,item_matrix)
        self.model = self.model.to(self.args.device)
        print(self.model)
        return self.model

    def get_scores_for_market(self, pred_file):
        # prepare for val set
        ref_path_val = os.path.join('./DATA/', self.args.tgt_market, 'valid_qrel.tsv')
        my_valid_qrel = read_qrel_file(ref_path_val)
        for key in pred_file.keys():
            if key not in my_valid_qrel:
                del my_valid_qrel[key]
        task_ov_val, task_ind_val = get_evaluations_final(pred_file, my_valid_qrel)

        return task_ov_val
    
    def fit(self, task_gen_all):
        opt = use_optimizer(self.model, self.config)
        loss_func = torch.nn.BCELoss()
        ############
        scores = ['ndcg_cut_10', 'recall_10']
        score_names = {
            'recall_10': {'val': 'r10_val', 'test': 'r10_test'},
            'ndcg_cut_10': {'val': 'ndcg10_val', 'test': 'ndcg10_test'}
        }
        ## Train
        ############
        self.model.train()
        best_result = 0
        best_count = 0
        for epoch in range(self.args.num_epoch):
            print('Epoch {} starts !'.format(epoch))
            total_loss = 0
            loss_lst = []

            train_tasksets = MetaMarket_Dataset(task_gen_all, num_negatives=self.args.num_negative, meta_split='train')
            train_dataloader = MetaMarket_DataLoader(train_tasksets, sample_batch_size=self.args.batch_size,
                                                     shuffle=True,
                                                     num_workers=3)
            # train the model for some certain iterations
            train_dataloader.refresh_dataloaders()
            data_lens = [len(train_dataloader[idx]) for idx in range(train_dataloader.num_tasks)]
            iteration_num = max(data_lens)
            for iteration in range(iteration_num):
                for subtask_num in range(train_dataloader.num_tasks): # get one batch from each dataloader
                    cur_train_dataloader = train_dataloader.get_iterator(subtask_num)
                    try:
                        train_user_ids, train_item_ids, train_targets = next(cur_train_dataloader)
                    except:
                        new_train_iterator = iter(train_dataloader[subtask_num])
                        train_user_ids, train_item_ids, train_targets = next(new_train_iterator)
                    
                    train_user_ids = train_user_ids.to(self.args.device)
                    train_item_ids = train_item_ids.to(self.args.device)
                    train_targets = train_targets.to(self.args.device)
                
                    opt.zero_grad()
                    ratings_pred = self.model(train_user_ids, train_item_ids)
                    loss = loss_func(ratings_pred.view(-1), train_targets)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                    loss_lst.append(loss.detach().cpu().data.numpy())
            
            sys.stdout.flush()
            print('-' * 80)
            valid_run_mf = self.predict(self.valid_dataloader, True)
            print('finish predicting!')
            print('loss:{}'.format(np.mean(loss_lst)))
            task_ov_val = self.get_scores_for_market(valid_run_mf)
            best_count +=1
            for score in scores:  # iterating over the scores
                score_val_name = score_names[score]['val']
                score_val = task_ov_val[score]
                print(
                    "======= Set val : score(" + score_val_name + ")=%0.12f =======" % score_val)
                if score_val_name == 'ndcg10_val' and score_val > best_result:
                    best_result = score_val
                    best_count = 0
                    print('Model is trained! and saved at:')
                    self.save()
            if best_count == 3:
                self.load(f'checkpoints/{self.args.tgt_market}_{self.args.src_markets}_{self.args.exp_name}.model')
                break


        
    # produce the ranking of items for users
    def predict(self, eval_dataloader, is_valid):
        self.model.eval()
        task_rec_all = []
        task_unq_users = set()
        for test_batch in eval_dataloader:
            test_user_ids, test_item_ids, test_targets = test_batch
    
            cur_users = [user.item() for user in test_user_ids]
            cur_items = [item.item() for item in test_item_ids]
            
            test_user_ids = test_user_ids.to(self.args.device)
            test_item_ids = test_item_ids.to(self.args.device)

            with torch.no_grad():
                batch_scores = self.model(test_user_ids, test_item_ids)
                batch_scores = batch_scores.detach().cpu().numpy()

            for index in range(len(test_user_ids)):
                task_rec_all.append((cur_users[index], cur_items[index], batch_scores[index][0].item()))

            task_unq_users = task_unq_users.union(set(cur_users))

        task_run_mf = get_run_mf(task_rec_all, task_unq_users, self.my_id_bank)
        return task_run_mf
    
    ## SAVE the model and idbank
    def save(self):
        if self.config['save_trained']:
            model_dir = f'checkpoints/{self.args.tgt_market}_{self.args.src_markets}_{self.args.exp_name}.model'
            cid_filename = f'checkpoints/{self.args.tgt_market}_{self.args.src_markets}_{self.args.exp_name}.pickle'
            print(f'--model: {model_dir}')
            print(f'--id_bank: {cid_filename}')
            torch.save(self.model.state_dict(), model_dir)
            with open(cid_filename, 'wb') as centralid_file:
                pickle.dump(self.my_id_bank, centralid_file)
    
    ## LOAD the model and idbank
    def load(self, checkpoint_dir):
        model_dir = checkpoint_dir
        state_dict = torch.load(model_dir, map_location=self.args.device)
        self.model.load_state_dict(state_dict, strict=False)
        print(f'Pretrained weights from {model_dir} are loaded!')

def numpy_to_torch(d, gpu=True):
    t = torch.from_numpy(d)
    if gpu and torch.cuda.device_count() > 0:
        t = t.cuda()
    return t

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape).coalesce()

def dropout_x(x, keep_prob):
        # 获取self.Graph中的大小，下标和值，Graph采用稀疏矩阵的表示方法SparseTensor
        size = x.size()
        index = x.indices().t()
        values = x.values()
        # 通过rand得到len(values)数量的随机数，加上keep_prob
        random_index = torch.rand(len(values)) + keep_prob
        # 通过对这些数字取int使得小于1的为0，在通过bool()将0->false,大于等于1的取True
        random_index = random_index.int().bool()
        # 利用上面得到的True，False数组选取下标，从而dropout了为False的下标
        index = index[random_index]
        # 由于dropout在训练和测试过程中的不一致，所以需要除以p
        values = values[random_index]/keep_prob
        # 得到新的graph
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

class LightGCN_no_w2v(torch.nn.Module):
    def __init__(self, config, graph):
        super(LightGCN_no_w2v, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.trainable_user = False
        self.trainable_item = False
        self.lighgcn = GNN(0.4, 3, self.latent_dim)
        self.graph = _convert_sp_mat_to_sp_tensor(graph).cuda()

        if config['embedding_user'] is None:
            self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
            self.trainable_user = True
        else:
            self.embedding_user = config['embedding_user']

        if config['embedding_item'] is None:
            self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
            self.trainable_item = True
        else:
            self.embedding_item = config['embedding_item']

        torch.nn.init.normal_(self.embedding_user.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.mlp_output = torch.nn.Linear(in_features=self.latent_dim*2, out_features=self.latent_dim*4)
        self.mlp_output1 = torch.nn.Linear(in_features=self.latent_dim*4, out_features=1)
        self.dropout_layer = torch.nn.Dropout(p=0.3)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        X = torch.cat([F.normalize(self.embedding_user.weight,dim=-1),F.normalize(self.embedding_item.weight,dim=-1)],dim=0)
        ui_embedding = F.normalize(self.lighgcn(self.graph, X),dim=-1)

        if self.trainable_user:
            user_embedding = ui_embedding[user_indices]
        else:
            user_embedding = self.embedding_user[user_indices]
        if self.trainable_item:
            item_embedding = ui_embedding[self.num_users+item_indices]
        else:
            item_embedding = self.embedding_item[item_indices]

        mf_vector = torch.mul(user_embedding, item_embedding)
        cat_vector = torch.cat([user_embedding, item_embedding], dim=-1)
        logits = self.affine_output(self.dropout_layer(mf_vector))
        mlp_vector = self.mlp_output1(self.dropout_layer(self.mlp_output(cat_vector).relu()))
        rating = self.logistic(logits+mlp_vector)
        return rating

    def init_weight(self):
        pass

class LightGCN_with_w2v(torch.nn.Module):
    def __init__(self, config, graph, user_matrix, item_matrix):
        super(LightGCN_with_w2v, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.trainable_user = False
        self.trainable_item = False
        self.lighgcn = GNN(0.4, 3, self.latent_dim)
        self.graph = _convert_sp_mat_to_sp_tensor(graph).cuda()
        self.user_matrix = torch.from_numpy(np.array(user_matrix)).float().cuda()
        self.item_matrix = torch.from_numpy(np.array(item_matrix)).float().cuda()

        if config['embedding_user'] is None:
            self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
            self.trainable_user = True
        else:
            self.embedding_user = config['embedding_user']

        if config['embedding_item'] is None:
            self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
            self.trainable_item = True
        else:
            self.embedding_item = config['embedding_item']

        torch.nn.init.normal_(self.embedding_user.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.mlp_output = torch.nn.Linear(in_features=self.latent_dim*2, out_features=self.latent_dim*4)
        self.mlp_output1 = torch.nn.Linear(in_features=self.latent_dim*4, out_features=1)
        self.dropout_layer = torch.nn.Dropout(p=0.3)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        X = torch.cat([F.normalize(self.embedding_user.weight,dim=-1),F.normalize(self.embedding_item.weight,dim=-1)],dim=0)
        ui_embedding = F.normalize(self.lighgcn(self.graph, X),dim=-1)

        if self.trainable_user:
            user_embedding = ui_embedding[user_indices]
        else:
            user_embedding = self.embedding_user[user_indices]
        if self.trainable_item:
            item_embedding = ui_embedding[self.num_users+item_indices]
        else:
            item_embedding = self.embedding_item[item_indices]

        user_embedding = F.normalize(user_embedding+self.user_matrix[user_indices],dim=-1)
        item_embedding = F.normalize(item_embedding+self.item_matrix[item_indices],dim=-1)

        mf_vector = torch.mul(user_embedding, item_embedding)
        cat_vector = torch.cat([user_embedding, item_embedding], dim=-1)
        logits = self.affine_output(self.dropout_layer(mf_vector))
        mlp_vector = self.mlp_output1(self.dropout_layer(self.mlp_output(cat_vector).relu()))
        rating = self.logistic(logits+mlp_vector)
        return rating

    def init_weight(self):
        pass

class GNN(torch.nn.Module):
    def __init__(self, dropout, layer_num, emb_size):
        super(GNN, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_num = layer_num
        self.emb_size = emb_size

    def forward(self, graph, X):
        ego1 = X
        all1 = [X]
        for i in range(self.layer_num):
            if self.training:
                graph_drop = dropout_x(graph, 0.6)
            else:
                graph_drop = graph
            agg = torch.sparse.mm(graph_drop,self.dropout(ego1))
            ego1 = agg+agg*self.dropout(ego1)
            all1.append(ego1)

        embs = torch.stack(all1,dim=1)
        light_out = torch.mean(embs, dim=1)
        return self.dropout(light_out)