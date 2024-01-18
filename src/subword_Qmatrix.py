import numpy as np
import pandas as pd
from scipy import sparse
import argparse
import os
import sentencepiece as spm
import glob
# import json


parser = argparse.ArgumentParser(description='Prepare datasets.')
parser.add_argument('--dataset', type=str, nargs='?', default='tagetomo') #tagetomo duolingo_hlr
parser.add_argument('--min_interactions', type=int, nargs='?', default=0)
parser.add_argument('--remove_nan_skills', type=bool, nargs='?', const=True, default=False)

parser.add_argument('--continuous_correct', type=bool, nargs='?', const=True, default=True)
parser.add_argument('--tags', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--lemma', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--subword_skills', type=bool, nargs='?', const=True, default=True)
parser.add_argument('--tokenizer_dir', type=str, nargs='?', default='./vocab/')
parser.add_argument('--vocab_size', type=int, nargs='?', default=5000)
parser.add_argument('--nbest', type=int, nargs='?', default=2)
options = parser.parse_args()



lang_tokenizers = {}


if options.subword_skills:
    print("Loading sentence piece tokenizers...", flush=True)
    for path in glob.glob("**" + "_vocab=" + str(options.vocab_size) + "**.model"):
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(path)
        if options.dataset == 'duolingo_hlr':
            lang = path.split("/")[-1].split("_")[0]
        else:
            lang = 'en1900'
            print('lang = en1900')
        lang_tokenizers[lang] = sp_model
        print(lang)

def extract_subword_skills_tage(word_df, nbest=1, tags=False, include_lemma=False):
    skill_df = []
    for i in range(len(word_df)):
        word = word_df[i]
        lang = 'en1900'
        nbest_segmentations = lang_tokenizers[lang].NBestEncodeAsPieces(word, nbest)

        piece_set = set()
        for segmentation in nbest_segmentations:
            if word not in segmentation:
                piece_set.update(segmentation)

        piece_set.add(word)

        print(piece_set)

        # piece_set.add("▁" + word_df[i].split("/")[1].split("<")[0])
        skill_set = "~~".join(map(lambda x: lang + ":" + x, piece_set))

        skill_df.append(skill_set)

    return skill_df



def prepare_tagetomo(min_interactions_per_user, remove_nan_skills, subword_skills=False, tags=False, lemma=False, drop_duplicates=False):

    # df = pd.read_csv("..\data\item.csv", sep='\t')
    df = pd.read_csv("..\data\item_subword_example1.csv", sep=',')



    if subword_skills:
        df["skill_id"] = extract_subword_skills_tage(df["word"], nbest=options.nbest, include_lemma=lemma, tags=tags)
    else:
        df["skill_id"] = df["word"]

    df["item_id"] = df["word_number"]-1


    if remove_nan_skills:
        df = df[~df["skill_id"].isnull()]

    else:
        df.loc[df["skill_id"].isnull(), "skill_id"] = "NaN"

    # Create list of KCs
    listOfKC = []
    for kc_raw in df["skill_id"].unique():
        for elt in kc_raw.split('~~'):
            listOfKC.append(elt)
    listOfKC = np.unique(listOfKC)
    print("number of skills:", len(listOfKC), flush=True)

    dict1_kc = {}
    dict2_kc = {}
    for k, v in enumerate(listOfKC):
        dict1_kc[v] = k
        dict2_kc[k] = v

    # df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    # unique_items, df["item_id"] = np.unique(df["item_id"], return_inverse=True)

    # df.reset_index(inplace=True, drop=True)  # Add unique identifier of the row
    # df["inter_id"] = df.index

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(listOfKC)))
    item_skill = np.array(df[["item_id", "skill_id"]])
    for i in range(len(item_skill)):
        splitted_kc = item_skill[i, 1].split('~~')
        for kc in splitted_kc:
            Q_mat[item_skill[i, 0], dict1_kc[kc]] = 1

    # Save data
    # modifier = ""
    # if subword_skills:
    #     submodifier = ""
    #     if tags:
    #         submodifier += "_tags"
    #     if lemma:
    #         submodifier += "_lemma"
    #     modifier = "_subword"+submodifier+"_vocab="+str(options.vocab_size)+"_nbest="+str(options.nbest)
    # elif tags:
    #     modifier = "_tags"
    #     if lemma:
    #         modifier += "_lemma"
    # elif lemma:
    #     modifier = "_lemma"


    # np.save("q_mat.npy", np.array(Q_mat))



    # with open("skill_map_word.csv", "w", encoding='utf-8') as skill_map_file:
    #     for k, v in enumerate(listOfKC):
    #         skill_map_file.write(str(k) + "\t" + v + "\n")

    # with open("E:/project/src/das3h/data/tagetomo/item_map"+modifier+".csv", "w", encoding='utf-8') as item_map_file:
    #     for k, v in enumerate(unique_items):
    #         item_map_file.write(str(k) + "\t" + v + "\n")

    return df, Q_mat





if __name__ == "__main__":




    df, Q_mat = prepare_tagetomo(min_interactions_per_user=options.min_interactions,
                                         remove_nan_skills=options.remove_nan_skills,
                                         tags=options.tags,
                                         lemma=options.lemma,
                                         subword_skills=options.subword_skills)


    print(Q_mat)
    print(Q_mat.shape)
    # np.set_printoptions(threshold=np.inf)
    print(Q_mat[0])
    print(np.count_nonzero(Q_mat[5]))

