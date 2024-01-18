import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
data_df = pd.read_csv('../data/interactions_features_minus.csv')

# df['format'] = df.apply(lambda x: x.topicId.split('-')[1], axis=1)
#
# df.to_csv('../data/dataset_all.csv', index=False, encoding='utf-8-sig')
# print(df.head(5))

# df3 = data_df.groupby(['format', 'result'])['result'].count()
# print(df3)


df1 = data_df.groupby(['userId']).count()

df2 = data_df.groupby(['format']).count()

print(data_df['result'].value_counts())

for format, format_df in data_df.groupby('format'):
    df3 = format_df.groupby(['userId']).count()
    df3['format'].hist(color='mediumpurple', grid=False)
    plt.show()

df1['format'].hist(color='mediumpurple', grid=False)
plt.show()


#
# df2.hist()
# plt.show()


