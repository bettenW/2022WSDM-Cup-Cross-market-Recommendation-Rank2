import argparse
import pandas as pd
import torch
from sklearn.model_selection import KFold

import os
import sys

sys.path.insert(1, 'src')
from model import Model
from utils import *
from data import *




def create_arg_parser():
    """Create argument parser for our baseline. """
    parser = argparse.ArgumentParser('LightGCN')
    
    # DATA  Arguments
    parser.add_argument('--data_dir', help='dataset directory', type=str, default='DATA/')
    parser.add_argument('--tgt_market', help='specify a target market name', type=str, default='t1') 
    parser.add_argument('--src_markets', help='specify none ("none") or a few source markets ("-" seperated) to augment the data for training', type=str, default='s1-s2') 
    
    parser.add_argument('--tgt_market_valid', help='specify validation run file for target market', type=str, default='DATA/t1/valid_run.tsv')
    parser.add_argument('--tgt_market_valid_pos', help='specify positive validation run file for target market', type=str,
                        default='DATA/t1/valid_qrel.tsv')
    parser.add_argument('--tgt_market_test', help='specify test run file for target market', type=str, default='DATA/t1/test_run.tsv') 
    
    parser.add_argument('--exp_name', help='name the experiment',type=str, default='baseline_toy')
    parser.add_argument('--model_name', help='name of the model', type=str, default='LightGCN_no_w2v')
    
    parser.add_argument('--train_data_file', help='the file name of the train data',type=str, default='train_merge.tsv') #'train.tsv' for the original data loading

    # MODEL arguments 
    parser.add_argument('--num_epoch', type=int, default=25, help='number of epoches')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--l2_reg', type=float, default=1e-07, help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent dimensions')
    parser.add_argument('--num_negative', type=int, default=99, help='num of negative samples during training')
    
    parser.add_argument('--cuda', action='store_true', help='use of cuda')
    parser.add_argument('--seed', type=int, default=20211120, help='manual seed init')
    
    return parser



def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    set_seed(args)
    
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.set_device(0)
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f'Running experiment on device: {args.device}')

    ############
    ## Target Market data
    ############
    train_file_names = args.train_data_file # 'train_5core.tsv', 'train.tsv' for the original data loading
    
    tgt_train_data_dir = os.path.join(args.data_dir, args.tgt_market, train_file_names)
    print(f'Loading target market {args.tgt_market}: {tgt_train_data_dir}')

    valid_train_ratings = pd.read_csv(args.tgt_market_valid, sep='\t')

    valid_run_mf = {}
    test_run_mf = {}

    if 'with' in args.model_name:
        random_list = [1, 2, 3, 4, 5, 6]
    else:
        random_list = [1,10,100,1000,2020,2025]

    split = 10
    for random_seed in random_list:
        kf = KFold(n_splits=split, shuffle=True, random_state=random_seed)

        for train_index, test_index in kf.split(valid_train_ratings):
            tgt_train_ratings = pd.read_csv(tgt_train_data_dir, sep='\t')
            valid_pos_ratings = pd.read_csv(args.tgt_market_valid_pos, sep='\t')

            X_train, X_valid = valid_train_ratings.iloc[train_index], valid_train_ratings.iloc[test_index]
            X_train.columns = ['userId','itemIds']
            X_valid.columns = ['userId','itemIds']
            uir = []
            for line in X_train.itertuples():
                user_id = line.userId
                item_ids = line.itemIds.strip().split(',')
                for cindex, item_id in enumerate(item_ids):
                    uir.append([user_id, item_id, 0])
            valid_all = pd.DataFrame(uir, columns=['userId', 'itemId', 'rating'])
            my_id_bank = Central_ID_Bank()
            tgt_task_generator = TaskGenerator(tgt_train_ratings, my_id_bank, valid_all, valid_pos_ratings, args)
            print('Loaded target data!\n')

            # task_gen_all: contains data for all training markets, index 0 for target market data
            task_gen_all = {
                0: tgt_task_generator
            }

            ############
            ## Validation and Test Run
            ############
            tgt_valid_dataloader = tgt_task_generator.instance_a_market_valid_dataloader(X_valid, args.batch_size, num_workers=3)
            tgt_test_dataloader = tgt_task_generator.instance_a_market_test_dataloader(args.tgt_market_test, args.batch_size, num_workers=3)

            ############
            ## Model
            ############
            if 'with' in args.model_name:
                user_matrix = tgt_task_generator.user_matrix+[[0.0]*tgt_task_generator.v_size for i in range(int(my_id_bank.last_user_index+1)-len(tgt_task_generator.user_matrix))]
                item_matrix = tgt_task_generator.item_matrix+[[0.0]*tgt_task_generator.v_size for i in range(int(my_id_bank.last_item_index+1)-len(tgt_task_generator.item_matrix))]
            else:
                user_matrix = None
                item_matrix = None

            graph = tgt_task_generator.instance_a_graph(int(my_id_bank.last_user_index+1), int(my_id_bank.last_item_index+1))
            mymodel = Model(args, my_id_bank,tgt_valid_dataloader,graph,user_matrix, item_matrix)
            mymodel.fit(task_gen_all)
            # validation data prediction
            b = mymodel.predict(tgt_valid_dataloader, True)
            for key,value in b.items():
                if key not in valid_run_mf:
                    valid_run_mf[key] = {}
                for key1,value1 in value.items():
                    if key1 not in valid_run_mf[key]:
                        valid_run_mf[key][key1] = 0
                    valid_run_mf[key][key1] += value1/len(random_list)

            a = mymodel.predict(tgt_test_dataloader, False)
            for key,value in a.items():
                if key not in test_run_mf:
                    test_run_mf[key] = {}
                for key1,value1 in value.items():
                    if key1 not in test_run_mf[key]:
                        test_run_mf[key][key1] = 0
                    test_run_mf[key][key1] += value1/(len(random_list)*split)

    print('Run output files:')

    if not os.path.exists("./merge"):
        os.mkdir("./merge")
    if not os.path.exists("./merge/{}".format(args.tgt_market)):
        os.mkdir("./merge/{}".format(args.tgt_market))

    valid_output_file = "./merge/{}/{}_valid.tsv".format(args.tgt_market,args.model_name)
    print(f'--validation: {valid_output_file}')
    write_run_file(valid_run_mf, valid_output_file)

    # test data prediction
    test_output_file = "./merge/{}/{}_test.tsv".format(args.tgt_market,args.model_name)
    print(f'--test: {test_output_file}')
    write_run_file(test_run_mf, test_output_file)
    
    print('Experiment finished successfully!')
    
if __name__=="__main__":
    main()