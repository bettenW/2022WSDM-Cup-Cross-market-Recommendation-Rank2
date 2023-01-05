import pandas as pd

t1_train = pd.read_csv("./DATA/t1/train.tsv", sep='\t')
print(len(t1_train))
t1_train_5core = pd.read_csv("./DATA/t1/train_5core.tsv", sep='\t')
print(len(t1_train_5core))
t1_train_5core["rating"] = t1_train_5core["rating"].apply(lambda x: 5.0)
final_train = pd.concat([t1_train,t1_train_5core],axis=0)
print(len(final_train))
final_train1 = final_train.drop_duplicates(subset=['userId','itemId'],keep='last')
print(len(final_train1))
final_train1.to_csv(path_or_buf="./DATA/t1/train_merge.tsv",sep='\t',columns=['userId','itemId','rating'])

t2_train = pd.read_csv("./DATA/t2/train.tsv", sep='\t')
print(len(t2_train))
t2_train_5core = pd.read_csv("./DATA/t2/train_5core.tsv", sep='\t')
print(len(t2_train_5core))
t2_train_5core["rating"] = t2_train_5core["rating"].apply(lambda x: 5.0)
final_train = pd.concat([t2_train,t2_train_5core],axis=0)
print(len(final_train))
final_train1 = final_train.drop_duplicates(subset=['userId','itemId'],keep='last')
print(len(final_train1))
final_train1.to_csv(path_or_buf="./DATA/t2/train_merge.tsv",sep='\t',columns=['userId','itemId','rating'])







