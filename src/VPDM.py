# coding: utf-8
import logging
import model
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np



embeddings_file='..\q_mat_word.npy'

if embeddings_file is not None:
    print("Using pretrained embeddings.", flush=True)
    embeddings = torch.FloatTensor(np.load(embeddings_file))
    print("embeddings shape:", np.shape(embeddings), flush=True)
    word, embed_dim = np.shape(embeddings)
    print(embeddings)


df = pd.read_csv("..\..\data\data_real_test.csv")
df_out = pd.read_csv("..\..\data\parameter_input.csv")
print(df.max())

df_item = df[['qId', 'format', 'word_Id']]

df_item = df_item.drop_duplicates(subset=['qId'])
df_item = df_item.reset_index(drop=True)

print(df_item.max())

train_data, test_data = train_test_split(df, test_size=0.2, random_state=19)
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
out_data = df_out.reset_index(drop=True)


print('data_loaded')

item2knowledge = {}
knowledge_set = set()
for i, s in df_item.iterrows():
    item_id, knowledge_codes = s['qId'], list(set([s['format']]))
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes)

print('knowledge_encoded')


batch_size = 256
user_n = 2014
item_n = 9500
word_n = 1900

knowledge_n = np.max(list(knowledge_set))+1



print('#knowledge_n =', knowledge_n)


def transform(user, item, word, format,section,wordlen,cefr, item2knowledge, score, batch_size):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64),
        torch.tensor(item, dtype=torch.int64),
        torch.tensor(word, dtype=torch.int64),
        torch.tensor(format, dtype=torch.int64),
        torch.tensor(section, dtype=torch.int64),
        torch.tensor(wordlen, dtype=torch.int64),
        torch.tensor(cefr, dtype=torch.int64),

        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

print('dataset transform...')

train_set, test_set, out_set= [
    transform(data["userId"], data["qId"], data["word_Id"], data["format"], data["section"],data["len"],data["CEFRId"],item2knowledge, data["result"], batch_size)
    for data in [train_data,  test_data, out_data]
]

print('model_MCDM')

logging.getLogger().setLevel(logging.INFO)

cdm = NCDM(knowledge_n, item_n, word_n, user_n, pretrained_embeddings=embeddings)
cdm.train(train_set,test_set,test_set, epoch=50, device="cuda")

# cdm.save("ncdm.snapshot")
#
# cdm.load("ncdm.snapshot")
