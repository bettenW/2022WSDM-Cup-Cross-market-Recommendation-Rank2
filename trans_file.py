import pandas as pd


a_t1_valid = pd.read_csv("./merge/result_6726/t1/valid_pred.tsv", sep='\t')
b_t1_valid = pd.read_csv("./merge/result_6716/t1/valid_pred.tsv", sep='\t')
c_t1_valid = pd.read_csv("./merge/result_dota_both_6718/t1/valid_pred.tsv", sep='\t')

a_t1_valid = a_t1_valid.merge(b_t1_valid,on=['userId','itemId'],how='left')
a_t1_valid = a_t1_valid.merge(c_t1_valid,on=['userId','itemId'],how='left')

a_t1_valid['merge_score'] = a_t1_valid['score_x']*1.5+a_t1_valid['score_y']*0.8+a_t1_valid['score']*1.2
a_t1_valid['merge_score'] = a_t1_valid['merge_score'].apply(lambda x: (x+100.0)*50+2.912)
a = a_t1_valid[['userId','itemId','merge_score']]
a.columns=['userId','itemId','score']
a.to_csv(path_or_buf="./submission/t1/valid_pred.tsv",sep='\t',columns=['userId','itemId','score'],index=False)

a_t2_valid = pd.read_csv("./merge/result_6726/t2/valid_pred.tsv", sep='\t')
b_t2_valid = pd.read_csv("./merge/result_6716/t2/valid_pred.tsv", sep='\t')
c_t2_valid = pd.read_csv("./merge/result_dota_both_6718/t2/valid_pred.tsv", sep='\t')

a_t2_valid = a_t2_valid.merge(b_t2_valid,on=['userId','itemId'],how='left')
a_t2_valid = a_t2_valid.merge(c_t2_valid,on=['userId','itemId'],how='left')

a_t2_valid['merge_score'] = a_t2_valid['score_x']*1.7+a_t2_valid['score_y']*0.8+a_t2_valid['score']*1.2
a_t2_valid['merge_score'] = a_t2_valid['merge_score'].apply(lambda x: (x+100.0)*50+1.1027)
a = a_t2_valid[['userId','itemId','merge_score']]
a.columns=['userId','itemId','score']
a.to_csv(path_or_buf="./submission/t2/valid_pred.tsv",sep='\t',columns=['userId','itemId','score'],index=False)

a_t1_test = pd.read_csv("./merge/result_6716/t1/test_pred.tsv", sep='\t')
a_t1_test.columns = ['userId','itemId','score_a']
b_t1_test = pd.read_csv("./merge/submission/t1/test_pred.tsv", sep='\t')
b_t1_test['score'] = b_t1_test['score'].apply(lambda x: (x-100.0)/50)
b_t1_test.columns = ['userId','itemId','score_b']
c_t1_test = pd.read_csv("./merge/result_dota_both_6718/t1/test_pred.tsv", sep='\t')
c_t1_test.columns = ['userId','itemId','score_c']
d_t1_test = pd.read_csv("./merge/result_cat_new/t1/test_pred.tsv", sep='\t')
d_t1_test.columns = ['userId','itemId','score_d']
e_t1_test = pd.read_csv("./merge/result_xgb_new/t1/test_pred.tsv", sep='\t')
e_t1_test.columns = ['userId','itemId','score_e']
f_t1_test = pd.read_csv("./merge/result_6726/t1/test_pred.tsv", sep='\t')
f_t1_test.columns = ['userId','itemId','score_f']

a_t1_test = a_t1_test.merge(b_t1_test,on=['userId','itemId'],how='left')
a_t1_test = a_t1_test.merge(c_t1_test,on=['userId','itemId'],how='left')
a_t1_test = a_t1_test.merge(d_t1_test,on=['userId','itemId'],how='left')
a_t1_test = a_t1_test.merge(e_t1_test,on=['userId','itemId'],how='left')
a_t1_test = a_t1_test.merge(f_t1_test,on=['userId','itemId'],how='left')

a_t1_test['merge_score'] = a_t1_test['score_a']*0.3+a_t1_test['score_c']*0.1*0.7+a_t1_test['score_d']*0.1+a_t1_test['score_e']*0.2+a_t1_test['score_f']*0.1*0.2
a_t1_test['merge_score'] = a_t1_test['merge_score'].apply(lambda x: (x+100.0)*50+1.7768687)
a = a_t1_test[['userId','itemId','merge_score']]
a.columns=['userId','itemId','score']
a.to_csv(path_or_buf="./submission/t1/test_pred.tsv",sep='\t',columns=['userId','itemId','score'],index=False)


a_t2_test = pd.read_csv("./merge/result_6716/t2/test_pred.tsv", sep='\t')
a_t2_test.columns = ['userId','itemId','score_a']
b_t2_test = pd.read_csv("./merge/submission/t2/test_pred.tsv", sep='\t')
b_t2_test['score'] = b_t2_test['score'].apply(lambda x: (x-100.0)/50)
b_t2_test.columns = ['userId','itemId','score_b']
c_t2_test = pd.read_csv("./merge/result_dota_both_6718/t2/test_pred.tsv", sep='\t')
c_t2_test.columns = ['userId','itemId','score_c']
d_t2_test = pd.read_csv("./merge/result_cat_new/t2/test_pred.tsv", sep='\t')
d_t2_test.columns = ['userId','itemId','score_d']
e_t2_test = pd.read_csv("./merge/result_xgb_new/t2/test_pred.tsv", sep='\t')
e_t2_test.columns = ['userId','itemId','score_e']
f_t2_test = pd.read_csv("./merge/result_6726/t2/test_pred.tsv", sep='\t')
f_t2_test.columns = ['userId','itemId','score_f']

a_t2_test = a_t2_test.merge(b_t2_test,on=['userId','itemId'],how='left')
a_t2_test = a_t2_test.merge(c_t2_test,on=['userId','itemId'],how='left')
a_t2_test = a_t2_test.merge(d_t2_test,on=['userId','itemId'],how='left')
a_t2_test = a_t2_test.merge(e_t2_test,on=['userId','itemId'],how='left')
a_t2_test = a_t2_test.merge(f_t2_test,on=['userId','itemId'],how='left')

a_t2_test['merge_score'] = a_t2_test['score_a']*0.2+a_t2_test['score_d']*0.2+a_t2_test['score_e']*0.2+a_t2_test['score_f']*0.1*0.9
a_t2_test['merge_score'] = a_t2_test['merge_score'].apply(lambda x: (x+100.0)*50+37.98765)
a = a_t2_test[['userId','itemId','merge_score']]
a.columns=['userId','itemId','score']
a.to_csv(path_or_buf="./submission/t2/test_pred.tsv",sep='\t',columns=['userId','itemId','score'],index=False)

