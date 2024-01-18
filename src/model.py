# coding: utf-8

import logging
import torch
import pandas
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from EduCDM import CDM
from sklearn import metrics



def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """

    device = torch.device("cuda")
    return torch.eye(num_classes, device=device)[y]


def linskill(y):
    """ encodes a tensor """
    device = torch.device("cuda")

    kc = torch.tensor([[1, 0, 0, 0],[1, 0, 0, 0],[1, 1, 0, 0],[1, 0, 1, 0],[1, 0, 0, 1]], device=device)

    return  kc[y]


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, word_n, student_n):
        self.knowledge_dim = 1900
        self.exer_n = exer_n
        self.emb_num = student_n
        self.word_n = word_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.word_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.word_n, 1)

        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, inut_word, inut_format, inut_section, inut_wordlen, inut_cefr, input_knowledge_point):
        # before prednet
        skill_onehot = to_categorical(inut_word, 1900)
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(inut_word))
        e_difficulty = torch.sigmoid(self.e_difficulty(inut_word))
        # prednet
        # input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = e_difficulty * (stat_emb - k_difficulty)* skill_onehot
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


class Net_nolinear(nn.Module):

    def __init__(self, knowledge_n, exer_n, word_n, student_n):

        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.word_n = word_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net_nolinear, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, 1)
        self.k_difficulty = nn.Embedding(self.exer_n, 1)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)


        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, inut_word, inut_format, inut_section, inut_wordlen, inut_cefr, input_knowledge_point):
        # before prednet
        # stu_emb = self.student_emb(stu_id)
        # stat_emb = torch.sigmoid(stu_emb)
        # k_difficulty = torch.sigmoid(self.k_difficulty(inut_word))
        # e_difficulty = torch.sigmoid(self.e_difficulty(inut_word))
        # # prednet
        # input_x = e_difficulty * (stat_emb - k_difficulty)

        stat_emb = 8 * (torch.sigmoid(self.student_emb(stu_id)) - 0.5)
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise)) * 2
        k_difficulty = 8 * (torch.sigmoid(self.k_difficulty(input_exercise)) - 0.5)

        input_x = torch.exp(-1.7 * e_difficulty * (stat_emb - k_difficulty))
        output_1 = torch.sigmoid(input_x)
        return output_1.view(-1)


class Net_dirt(nn.Module):

    def __init__(self, knowledge_n, exer_n, word_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.word_n = word_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net_dirt, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.word_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.word_n, self.knowledge_dim)
        self.k_difficulty_i = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty_i = nn.Embedding(self.exer_n, self.knowledge_dim)


        self.a = nn.Sequential(
            nn.Linear(self.prednet_input_len, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.b = nn.Sequential(
            nn.Linear(self.prednet_input_len, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.theta = nn.Sequential(
            nn.Linear(self.knowledge_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise,inut_word, inut_format,inut_section,inut_wordlen,inut_cefr, input_knowledge_point):

        format_onehot = to_categorical(inut_format, 5)
        section_onehot = to_categorical(inut_section-1, 19)
        cefr_onehot = to_categorical(inut_cefr, 6)

        stu_emb = self.student_emb(stu_id)
        e_difficulty_emb = self.e_difficulty(inut_word)
        k_difficulty_emb = self.k_difficulty(inut_word)

        e_difficulty_i_emb = self.e_difficulty_i(input_exercise)
        k_difficulty_i_emb = self.k_difficulty_i(input_exercise)

        stat_emb = 8 * (torch.sigmoid(self.theta(stu_emb)) - 0.5)

        # print(inut_section.size())
        # print(section_onehot.size())
        # print(inut_wordlen.size())
        # print(torch.unsqueeze(inut_wordlen,1).size())
        e_inut = torch.cat([e_difficulty_emb], dim=1)
        k_inut = torch.cat([k_difficulty_emb], dim=1)

        e_difficulty =  torch.sigmoid(self.a(e_inut))*2
        k_difficulty = 8 * (torch.sigmoid(self.b(k_inut)) - 0.5)

        input_x = torch.exp(-1.7 * e_difficulty * (stat_emb - k_difficulty))
        output_1 = torch.sigmoid(input_x)
        return output_1.view(-1)


class Net_mdirt(nn.Module):

    def __init__(self, knowledge_n, exer_n, word_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.word_n = word_n
        self.stu_dim = self.knowledge_dim
        self.net_input_len = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net_mdirt, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.word_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.word_n, self.knowledge_dim)


        self.k_difficulty_i = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty_i = nn.Embedding(self.exer_n, self.knowledge_dim)


        self.a = nn.Sequential(
            nn.Linear(self.net_input_len, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.b = nn.Sequential(
            nn.Linear(self.net_input_len, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.knowledge_dim)
        )
        self.theta = nn.Sequential(
            nn.Linear(self.knowledge_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.knowledge_dim)
        )

        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise,inut_word, inut_format,inut_section,inut_wordlen,inut_cefr, input_knowledge_point):

        format_onehot = to_categorical(inut_format, 5)
        section_onehot = to_categorical(inut_section-1, 19)
        cefr_onehot = to_categorical(inut_cefr, 6)

        stu_emb = self.student_emb(stu_id)
        e_difficulty_emb = self.e_difficulty(inut_word)
        k_difficulty_emb = self.k_difficulty(inut_word)

        e_difficulty_i_emb = self.e_difficulty_i(input_exercise)
        k_difficulty_i_emb = self.k_difficulty_i(input_exercise)


        stat_emb = torch.sigmoid(self.theta(stu_emb))

        # print(inut_section.size())
        # print(section_onehot.size())
        # print(inut_wordlen.size())
        # print(torch.unsqueeze(inut_wordlen,1).size())
        e_inut = torch.cat([e_difficulty_i_emb], dim=1)
        k_inut = torch.cat([k_difficulty_i_emb], dim=1)


        e_difficulty =  torch.sigmoid(self.a(e_inut))
        k_difficulty = torch.sigmoid(self.b(k_inut))


        input_x = e_difficulty * (stat_emb - k_difficulty)
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        # input_x = torch.exp(-1.7 * e_difficulty * (stat_emb - k_difficulty))
        # output_1 = torch.sigmoid(input_x)


        return output_1.view(-1)


class Net_3pldirt(nn.Module):

    def __init__(self, knowledge_n, exer_n, word_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.word_n = word_n
        self.stu_dim = self.knowledge_dim
        self.net_input_len = self.knowledge_dim+5
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net_3pldirt, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.word_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.word_n, self.knowledge_dim)
        self.guess = nn.Embedding(self.word_n, self.knowledge_dim)

        self.k_difficulty_i = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty_i = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.guess_i = nn.Embedding(self.exer_n, self.knowledge_dim)

        self.a = nn.Sequential(
            nn.Linear(self.net_input_len, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.b = nn.Sequential(
            nn.Linear(self.net_input_len, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.knowledge_dim)
        )
        self.theta = nn.Sequential(
            nn.Linear(self.knowledge_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.knowledge_dim)
        )

        self.g = nn.Sequential(
            nn.Linear(self.net_input_len-1, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        self.guess_adjust = nn.Linear(1, 1)

        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, inut_word, inut_format, inut_section, inut_wordlen, inut_cefr,
                input_knowledge_point):

        format_onehot = to_categorical(inut_format, 5)
        section_onehot = to_categorical(inut_section - 1, 19)
        cefr_onehot = to_categorical(inut_cefr, 6)

        stu_emb = self.student_emb(stu_id)

        e_difficulty_emb = self.e_difficulty(inut_word)
        k_difficulty_emb = self.k_difficulty(inut_word)
        guess_emb = self.guess(inut_word)

        e_difficulty_i_emb = self.e_difficulty_i(input_exercise)
        k_difficulty_i_emb = self.k_difficulty_i(input_exercise)
        guess_i_emb = self.guess_i(input_exercise)



        stat_emb = torch.sigmoid(self.theta(stu_emb))
        # print(torch.unsqueeze(inut_wordlen,1).size())
        e_inut = torch.cat([e_difficulty_emb,format_onehot], dim=1)
        k_inut = torch.cat([k_difficulty_emb,format_onehot], dim=1)
        g_inut = torch.cat([guess_emb,stu_emb], dim=1)

        e_difficulty = torch.sigmoid(self.a(e_inut))
        k_difficulty = torch.sigmoid(self.b(k_inut))
        guess_ad = torch.sigmoid(self.g(g_inut))

        # guess = torch.sigmoid(self.guess_adjust(guess_ad))

        input_x = e_difficulty * (stat_emb - k_difficulty)
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_p = torch.sigmoid(self.prednet_full3(input_x))

        # input_x = torch.exp(-1.7 * e_difficulty * (stat_emb - k_difficulty))
        # output_1 = torch.sigmoid(input_x)

        output = guess_ad + (1 - guess_ad) * output_p

        return output.view(-1)


class Net_4pldirt(nn.Module):

    def __init__(self, knowledge_n, exer_n, word_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.word_n = word_n
        self.stu_dim = self.knowledge_dim
        self.net_input_len = self.knowledge_dim+5
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net_4pldirt, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.word_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.word_n, self.knowledge_dim)
        self.guess = nn.Embedding(self.word_n, self.knowledge_dim)
        self.slip = nn.Embedding(self.word_n, self.knowledge_dim)

        self.k_difficulty_i = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty_i = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.guess_i = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.slip_i = nn.Embedding(self.exer_n, self.knowledge_dim)

        self.a = nn.Sequential(
            nn.Linear(self.net_input_len, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.b = nn.Sequential(
            nn.Linear(self.net_input_len, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.knowledge_dim)
        )
        self.theta = nn.Sequential(
            nn.Linear(self.knowledge_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.knowledge_dim)
        )

        self.g = nn.Sequential(
            nn.Linear(self.net_input_len+4, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        self.s = nn.Sequential(
            nn.Linear(self.net_input_len+4, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        # self.guess_adjust = nn.Linear(1, 1)
        #
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, inut_word, inut_format, inut_section, inut_wordlen, inut_cefr,
                input_knowledge_point):

        format_onehot = to_categorical(inut_format, 5)
        section_onehot = to_categorical(inut_section - 1, 19)
        cefr_onehot = to_categorical(inut_cefr, 6)

        stu_emb = self.student_emb(stu_id)

        e_difficulty_emb = self.e_difficulty(inut_word)
        k_difficulty_emb = self.k_difficulty(inut_word)
        guess_emb = self.guess(inut_word)
        slip_emb = self.slip(inut_word)

        e_difficulty_i_emb = self.e_difficulty_i(input_exercise)
        k_difficulty_i_emb = self.k_difficulty_i(input_exercise)
        guess_i_emb = self.guess_i(input_exercise)
        slip_i_emb = self.slip_i(input_exercise)


        stat_emb = torch.sigmoid(self.theta(stu_emb))
        # print(torch.unsqueeze(inut_wordlen,1).size())
        e_inut = torch.cat([e_difficulty_emb,format_onehot], dim=1)
        k_inut = torch.cat([k_difficulty_emb,format_onehot], dim=1)
        g_inut = torch.cat([guess_emb,format_onehot, stu_emb], dim=1)
        s_inut = torch.cat([slip_emb, format_onehot, stu_emb], dim=1)


        e_difficulty = torch.sigmoid(self.a(e_inut))
        k_difficulty = torch.sigmoid(self.b(k_inut))
        guess_ad = torch.sigmoid(self.g(g_inut))
        slip_ad = torch.sigmoid(self.s(s_inut))

        # guess = torch.sigmoid(self.guess_adjust(guess_ad))

        input_x = e_difficulty * (stat_emb - k_difficulty)
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_p = torch.sigmoid(self.prednet_full3(input_x))

        # input_x = torch.exp(-1.7 * e_difficulty * (stat_emb - k_difficulty))
        # output_1 = torch.sigmoid(input_x)

        output = guess_ad + (slip_ad - guess_ad) * output_p

        return output.view(-1)


class Net_lsirt(nn.Module):

    def __init__(self, knowledge_n, exer_n, word_n, student_n, pretrained_embeddings, freeze_pretrained=True):
        self.knowledge_dim = 5 #1900 #1900 1742 3610
        self.exer_n = exer_n
        self.emb_num = student_n
        self.word_n = word_n
        self.stu_dim = self.knowledge_dim
        self.net_input_len = self.knowledge_dim+5
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net_lsirt, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.word_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.word_n, self.knowledge_dim)
        self.guess = nn.Embedding(self.word_n, self.knowledge_dim)
        self.slip = nn.Embedding(self.word_n, self.knowledge_dim)

        # self.k_difficulty_i = nn.Embedding(self.exer_n, self.knowledge_dim)
        # self.e_difficulty_i = nn.Embedding(self.exer_n, self.knowledge_dim)
        # self.guess_i = nn.Embedding(self.exer_n, self.knowledge_dim)
        # self.slip_i = nn.Embedding(self.exer_n, self.knowledge_dim)

        self.a = nn.Sequential(
            PosLinear(self.net_input_len, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.b = nn.Sequential(
            PosLinear(self.net_input_len, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.knowledge_dim)
        )

        self.theta = nn.Sequential(
            PosLinear(self.knowledge_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.knowledge_dim)
        )

        self.g = nn.Sequential(
            PosLinear(self.net_input_len+self.knowledge_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        self.s = nn.Sequential(
            PosLinear(self.net_input_len+self.knowledge_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        # self.guess_adjust = nn.Linear(1, 1)
        #
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.2)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.2)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        if pretrained_embeddings is not None:
            print("pretrained_embeddings")
            self.q_embed = nn.Embedding.from_pretrained(pretrained_embeddings)



    def forward(self, stu_id, input_exercise, inut_word, inut_format, inut_section, inut_wordlen, inut_cefr, input_knowledge_point):

        format_onehot = to_categorical(inut_format, 5)
        skill_onehot = to_categorical(inut_word,1900)
        lin_skill = linskill(inut_format)
        subword_skill = self.q_embed(inut_word)



        # section_onehot = to_categorical(inut_section - 1, 19)
        # cefr_onehot = to_categorical(inut_cefr, 6)

        stu_emb = self.student_emb(stu_id)

        e_difficulty_emb = self.e_difficulty(inut_word)
        k_difficulty_emb = self.k_difficulty(inut_word)
        guess_emb = self.guess(inut_word)
        slip_emb = self.slip(inut_word)

        # e_difficulty_i_emb = self.e_difficulty_i(input_exercise)
        # k_difficulty_i_emb = self.k_difficulty_i(input_exercise)
        # guess_i_emb = self.guess_i(input_exercise)
        # slip_i_emb = self.slip_i(input_exercise)


        stat_emb = torch.sigmoid(self.theta(stu_emb))
        #stat_emb = torch.sigmoid(stu_emb)

        # print(torch.unsqueeze(inut_wordlen,1).size())
        e_inut = torch.cat([e_difficulty_emb,format_onehot], dim=1)
        k_inut = torch.cat([k_difficulty_emb,format_onehot], dim=1)
        g_inut = torch.cat([guess_emb,format_onehot, stu_emb], dim=1)
        s_inut = torch.cat([slip_emb,format_onehot, stu_emb], dim=1)


        e_difficulty = torch.sigmoid(self.a(e_inut))
        k_difficulty = torch.sigmoid(self.b(k_inut))
        guess_ad = torch.sigmoid(self.g(g_inut))
        slip_ad = torch.sigmoid(self.s(s_inut))

        # guess = torch.sigmoid(self.guess_adjust(guess_ad))

        #input = e_difficulty * (stat_emb - k_difficulty)

        input_x = e_difficulty * (stat_emb - k_difficulty)*input_knowledge_point
        #lin_skill
        #print(input_knowledge_point)


        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_p = torch.sigmoid(self.prednet_full3(input_x))

        output = guess_ad + (slip_ad - guess_ad) * output_p
        #output = guess_ad + (1 - guess_ad) * output_p
        #output = output_p

        return output.view(-1) ,e_difficulty.view(-1) , k_difficulty.view(-1,5) , slip_ad.view(-1) , guess_ad.view(-1), input_exercise.view(-1),inut_word.view(-1), inut_format.view(-1),stu_id.view(-1),stat_emb.view(-1,5)

class VPDM(CDM):


    def __init__(self, knowledge_n, exer_n, word_n, student_n, pretrained_embeddings):
        super(VPDM, self).__init__()
        #self.ncdm_net =Net_lsirt(knowledge_n, exer_n, word_n, student_n)
        self.ncdm_net = Net_lsirt(knowledge_n, exer_n, word_n, student_n, pretrained_embeddings)

    def train(self, train_data, test_data=None,out_data=None, epoch=10, device="cuda", lr=0.003, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        patience = 3
        best_auc = None
        patience_used = 0
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, word_id, format_id, section_id,wordlen, cefr_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)

                word_id: torch.Tensor = word_id.to(device)
                format_id: torch.Tensor = format_id.to(device)
                section_id: torch.Tensor = section_id.to(device)
                wordlen: torch.Tensor = wordlen.to(device)
                cefr_id: torch.Tensor = cefr_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred, a, b, c,d,q,w,f,sid,s =self.ncdm_net(user_id, item_id, word_id, format_id,section_id,wordlen, cefr_id, knowledge_emb)

                pred: torch.Tensor= pred.to(device)

                loss = loss_function(pred, y)



                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            if test_data is not None:
                auc, accuracy, mae, rmse = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, mae: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy,mae,rmse ))


            if best_auc is None or auc >= best_auc:
                patience_used = 0
                best_auc = auc
            else:
                patience_used += 1
                if patience_used >= patience:
                    break

        auc, accuracy, mae, rmse = self.output_para(out_data, device=device)
        print("[Epoch %d] auc: %.6f, accuracy: %.6f, mae: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, mae, rmse))

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []

        a_p = []
        b_p = []
        c_p = []
        d_p = []
        q_p = []


        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, word_id, format_id,section_id,wordlen, cefr_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            word_id: torch.Tensor = word_id.to(device)
            format_id: torch.Tensor = format_id.to(device)
            section_id: torch.Tensor = section_id.to(device)
            wordlen: torch.Tensor = wordlen.to(device)
            cefr_id: torch.Tensor = cefr_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred, a, b, c,d,q,w,f,sid,s = self.ncdm_net(user_id, item_id, word_id, format_id, section_id, wordlen, cefr_id,
                                             knowledge_emb)

            pred: torch.Tensor = pred.to(device)
            y_pred.extend(pred.detach().cpu().tolist())


            y_true.extend(y.tolist())

        mae = metrics.mean_absolute_error(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), mae, rmse
    def output_para(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []

        a_p = []
        b_p = []
        c_p = []
        d_p = []
        q_p = []
        f_p = []
        w_p= []
        sid_p = []
        s_p = []

        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, word_id, format_id,section_id,wordlen, cefr_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            word_id: torch.Tensor = word_id.to(device)
            format_id: torch.Tensor = format_id.to(device)
            section_id: torch.Tensor = section_id.to(device)
            wordlen: torch.Tensor = wordlen.to(device)
            cefr_id: torch.Tensor = cefr_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred, a, b, c,d,q,w,f ,sid,s = self.ncdm_net(user_id, item_id, word_id, format_id, section_id, wordlen, cefr_id,
                                             knowledge_emb)

            pred: torch.Tensor = pred.to(device)
            a: torch.Tensor = a.to(device)
            b: torch.Tensor = b.to(device)
            c: torch.Tensor = c.to(device)
            d: torch.Tensor = d.to(device)
            q: torch.Tensor = q.to(device)
            w: torch.Tensor = w.to(device)
            f: torch.Tensor = f.to(device)
            sid: torch.Tensor = sid.to(device)
            s: torch.Tensor = s.to(device)


            y_pred.extend(pred.detach().cpu().tolist())
            a_p.extend(a.detach().cpu().tolist())
            b_p.extend(b.detach().cpu().tolist())
            c_p.extend(c.detach().cpu().tolist())
            d_p.extend(d.detach().cpu().tolist())
            q_p.extend(q.detach().cpu().tolist())
            w_p.extend(w.detach().cpu().tolist())
            f_p.extend(f.detach().cpu().tolist())
            sid_p.extend(sid.detach().cpu().tolist())
            s_p.extend(s.detach().cpu().tolist())


            # a_p += list(a.view(len(pred)).data.numpy())
            # b_p += list(b.view(len(pred)).data.numpy())
            # c_p += list(c.view(len(pred)).data.numpy())
            # d_p += list(d.view(len(pred)).data.numpy())
            # q += list(np.array(item_id))

            y_true.extend(y.tolist())

        d = {'userId': sid_p,'qId': q_p,'wordId': w_p,'format': f_p, 'a': a_p, 'b': b_p, 'c': c_p, 'd': d_p,'user_emb': s_p}
        # d = { 'a': a_p, 'b': b_p, 'c': c_p, 'd': d_p}
        print(len(q_p),len(a_p),len(b_p),len(c_p),len(d_p))
        df = pandas.DataFrame(d)
        df.to_csv('item_parameter_format_stuem.csv')
        print('saved!')

        mae = metrics.mean_absolute_error(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), mae, rmse

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)
