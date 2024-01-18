import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_df = pd.read_csv('../data/interactions_features.csv')

# df['format'] = df.apply(lambda x: x.topicId.split('-')[1], axis=1)
#
# df.to_csv('../data/dataset_all.csv', index=False, encoding='utf-8-sig')
# print(df.head(5))

# df2 = df.groupby(['format', 'result'])['result'].count()


data_df['word_Id'] = data_df['word_number'] -1


print(data_df['userId'].nunique())

print(len(data_df))
data_df.drop_duplicates(inplace=True)
print(len(data_df))

# Clean data
user_wise_lst = list()


for user, user_df in data_df.groupby('userId'):
    # if user_df['format'].nunique() >= 4 and len(user_df) >= 3500:
    if user <= 2013:
        user_wise_lst.append(user_df)


# np.random.seed(1)
# np.random.shuffle(user_wise_lst)
user_wise_df = pd.concat(user_wise_lst).reset_index(drop=True)


print(len(user_wise_df))
print(user_wise_df.head())

# Re-index
# skill2name = dict()
# new_user_id, new_item_id, new_format_id = dict(), dict(), dict()
# user_cnt, item_cnt = 0, 0
#
# for u_id, i_id in zip(user_wise_df['userId'].values, user_wise_df['qId'].values):
#     if u_id not in new_user_id:
#         new_user_id[u_id] = user_cnt
#         user_cnt += 1
#     if i_id not in new_item_id:
#         new_item_id[i_id] = item_cnt
#         item_cnt += 1
#
# new_format_id =  {1: 0, 2: 1, 5: 2, 9: 3, 10: 4}



# user_wise_df['userId'] = user_wise_df['userId'].apply(lambda x: new_user_id[x])
# user_wise_df['qId'] = user_wise_df['qId'].apply(lambda x: new_item_id[x])
# user_wise_df['format'] = user_wise_df['format'].apply(lambda x: new_format_id[x])


print(user_wise_df.head())

user_wise_df.drop_duplicates(inplace=True)
# print(len(user_wise_df))

user_wise_df.to_csv('../data/interactions_features_minus.csv', index=False, encoding='utf-8-sig')

# print(user_cnt, item_cnt)





