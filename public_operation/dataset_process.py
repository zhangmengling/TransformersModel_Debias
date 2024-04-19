import random

from sklearn.metrics import davies_bouldin_score
from datasets import load_dataset, load_metric

import scipy.cluster.hierarchy as shc
from matplotlib import pyplot
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
import re
from pandas.core.frame import DataFrame

from sklearn.model_selection import train_test_split


seed = 999
random.seed(seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("-->device", device)
torch.set_grad_enabled(True)

class IdentityDetect():
    def __init__(self):
        # term_file_path = "public_operation/identity_class.json"
        term_file_path = "public_operation/identity_term_replace.json"
        with open(term_file_path, "r") as term_file:
            self.term_class_dict = json.load(term_file)
        self.identities = list(self.term_class_dict.keys())

        self.all_identity_terms = []
        for identity in self.identities:
            self.all_identity_terms = self.all_identity_terms + self.term_class_dict[identity]

        self.identity_idi_map = {'male': ['female', 'homosexual'], 'female': ['male', 'homosexual'],
                            'homosexual': ['male', 'female'],
                            'christian': ['muslim', 'jewish'], 'muslim': ['christian', 'jewish'],
                            'jewish': ['christian', 'muslim'],
                            'black': ['white'], 'white': ['black']}

        self.characters = [',', ':', '.', '?', '!', '`', '@', '#', '$', '%', '^', '&', '*', '/', ';', '[', ']', '{', '}', '(', ')',
                           '-', '\\', '\'', '\"', '+', '=', '<', '>', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

        terms_f2m_path = "public_operation/identity_terms_f2m.json"
        terms_m2f_path = "public_operation/identity_terms_m2f.json"
        with open(terms_f2m_path, "r") as term_file:
            self.terms_f2m = json.load(term_file)
        with open(terms_m2f_path, "r") as term_file:
            self.terms_m2f = json.load(term_file)

        self.gender_terms = self.term_class_dict['male'] + self.term_class_dict['female'] + self.term_class_dict[
            'homosexual']
        self.religion_terms = self.term_class_dict['christian'] + self.term_class_dict['muslim'] + self.term_class_dict[
            'jewish']
        self.race_terms = self.term_class_dict['black'] + self.term_class_dict['white']

        self.identity_terms_map = {'male': self.gender_terms, 'female': self.gender_terms,
                                   'homosexual': self.gender_terms,
                                   'christian': self.religion_terms, 'muslim': self.religion_terms,
                                   'jewish': self.religion_terms,
                                   'black': self.race_terms, 'white': self.race_terms}

        # self.identity_category = [['male', 'female', 'homosexual'], ['christian', 'muslim', 'jewish'],
        #                           ['black', 'white']]
        self.identity_category = {"gender": ['male', 'female', 'homosexual'], "religion": ['christian', 'muslim', 'jewish'],
                                  "race": ['black', 'white']}


    def add_space(self, word):
        return ' ' + word + ' '

    def consider_characters(self, term):
        all_terms = [term]
        for cha in self.characters:
            all_terms.append(term + cha)
            all_terms.append(cha + term)
            all_terms.append(cha + term + cha)
            all_terms.append("'" + term + "'")
            all_terms.append(term + "s")
            all_terms.append(term + "'s")
        return all_terms

    def consider_characters1(self, term):
        all_terms = [term]
        all_terms.append(term + "s")
        return all_terms

    def mask_byDict(self, review, terms):
        '''
        mask_byDict: Mask terms in a text
        args
            review (str): Text
            terms (dict): tems. Mask kes by value.
        return tuple [(str) new masked review text, (dict) term occurances]
        '''
        review = review.lower()
        for word, initial in terms.items():
            # count_dict[word] = len(re.findall(add_space(word), review))
            review = review.replace(self.add_space(word), self.add_space(initial))

            extending_word = self.consider_characters(word)
            # if word == 'father':
            #     print("-->extending_word", extending_word)
            extending_initial = self.consider_characters(initial)
            for i in range(0, len(extending_word)):
                review = review.replace(self.add_space(extending_word[i]), self.add_space(extending_initial[i]))
        return review

    def identity_detect(self, text, identity):  # term_class
        # terms = self.term_class_dict[identity]
        # for term in terms:
        #     extending_terms = self.consider_characters(term)
        #     for t in extending_terms:
        #         if self.add_space(t) in text:
        #             return True
        # return False

        # pattern = r"[\",:.?!`@#$%^&*\/;\-%_+=><'\[\]{}()] "
        # pattern = self.characters
        # text = re.split(pattern, text)
        # text = re.split("[ ;.,:?!'/()\[\]*&`@#%$^{}]", text)
        text = re.split("[ ;.,:?!'/()\[\]*&`@#%$^{}_+=<>0123456789\"\-\s+]", text)
        if isinstance(identity, list):
            terms = identity
        else:
            terms = self.term_class_dict[identity]
        for term in terms:
            extending_terms = self.consider_characters1(term)
            for t in extending_terms:
                if t in text:
                    return True
        return False

    def identity_category_detect(self, text, category):
        """
        identity_category: e.g., gender, religion, race
        """
        identities = self.identity_category[category]
        exist = False
        for identity in identities:
            exist = self.identity_detect(text, identity)
            if exist == True:
                return exist
        return exist

    def which_identity(self, text):
        contain_identity = []
        for identity in self.identities:
            if self.identity_detect(text, identity) == True:
                contain_identity.append(identity)
        return contain_identity

    def replace_identity(self, text, identity_terms, replace_class):  # term_class
        text = text.lower()
        text = re.split("([ ;.,:?!'/()\[\]*&`@#%$^{}_+=<>0123456789\"\-\s+])", text)
        replace_text = text.copy()
        terms = identity_terms
        for term in terms:
            extending_terms = self.consider_characters1(term)
            for t in extending_terms:
                if t in replace_text:
                    for i in range(0, len(replace_text)):
                        if replace_text[i] == t:
                            replace_text[i] = random.choice(self.term_class_dict[replace_class])
        if replace_text == text:
            print("replace_text == text")
            # raise AttributeError("replace_text == text")
        str = ''
        replaced_text = str.join(replace_text)
        return replaced_text

    def generate_idi(self, text):
        # terms_f2m_path = "public_operation/identity_terms_f2m.json"
        # terms_m2f_path = "public_operation/identity_terms_m2f.json"
        # with open(terms_f2m_path, "r") as term_file:
        #     self.terms_f2m = json.load(term_file)
        # with open(terms_m2f_path, "r") as term_file:
        #     self.terms_m2f = json.load(term_file)
        #
        # self.gender_terms = self.term_class_dict['male'] + self.term_class_dict['female'] + self.term_class_dict[
        #     'homosexual']
        # self.religion_terms = self.term_class_dict['christian'] + self.term_class_dict['muslim'] + self.term_class_dict[
        #     'jewish']
        # self.race_terms = self.term_class_dict['black'] + self.term_class_dict['white']
        #
        # self.identity_terms_map = {'male': self.gender_terms, 'female': self.gender_terms,
        #                            'homosexual': self.gender_terms,
        #                            'christian': self.religion_terms, 'muslim': self.religion_terms,
        #                            'jewish': self.religion_terms,
        #                            'black': self.race_terms, 'white': self.race_terms}
        #
        # self.identity_category = [['male', 'female', 'homosexual'], ['christian', 'muslim', 'jewish'],
        #                           ['black', 'white']]

        idis = []
        identity_index = []
        for i in range(0, len(self.identities)):
            identity = self.identities[i]
            if self.identity_detect(text, identity) == True:
                identity_index.append(i)
        if len(identity_index) == 0:
            return False
        else:
            identity_idi_considered = []
            for index in identity_index:
                identity = self.identities[index]
                identity_idis = self.identity_idi_map[identity]
                identity_idis = [i for i in identity_idis if i not in identity_idi_considered]
                identity_idi_considered = identity_idi_considered + identity_idis
                for identity_idi in identity_idis:
                    if identity == 'female' and identity_idi == 'male':
                        new_text = self.mask_byDict(text, self.terms_f2m)
                        idis.append(new_text)
                    elif identity == 'male' and identity_idi == 'female':
                        new_text = self.mask_byDict(text, self.terms_m2f)
                        idis.append(new_text)
                    else:
                        # print("-->identity", identity)
                        # print("-->identity_idi", identity_idi)
                        new_text = self.replace_identity(text, self.identity_terms_map[identity], identity_idi)
                        if new_text != text:
                            idis.append(new_text)
        return idis

    def generate_gender_version(self, text):
        term_file_path = "public_operation/identity_term_replace.json"
        with open(term_file_path, "r") as term_file:
            self.term_class_dict = json.load(term_file)
        # self.identities = list(self.term_class_dict.keys())
        self.identities = ["male", "female"]

        self.all_identity_terms = []
        for identity in self.identities:
            self.all_identity_terms = self.all_identity_terms + self.term_class_dict[identity]

        self.identity_idi_map = {'male': ['female'], 'female': ['male']}

        terms_f2m_path = "public_operation/identity_terms_f2m.json"
        terms_m2f_path = "public_operation/identity_terms_m2f.json"
        with open(terms_f2m_path, "r") as term_file:
            self.terms_f2m = json.load(term_file)
        with open(terms_m2f_path, "r") as term_file:
            self.terms_m2f = json.load(term_file)

        self.gender_terms = self.term_class_dict['male'] + self.term_class_dict['female']

        self.identity_terms_map = {'male': self.gender_terms, 'female': self.gender_terms}

        self.identity_category = [['male', 'female']]

        idis = []
        identity_index = []
        for i in range(0, len(self.identities)):
            identity = self.identities[i]
            if self.identity_detect(text, identity) == True:
                identity_index.append(i)
        if len(identity_index) == 0:
            return False
        else:
            identity_idi_considered = []
            for index in identity_index:
                identity = self.identities[index]
                identity_idis = self.identity_idi_map[identity]
                identity_idis = [i for i in identity_idis if i not in identity_idi_considered]
                identity_idi_considered = identity_idi_considered + identity_idis
                for identity_idi in identity_idis:
                    if identity == 'female' and identity_idi == 'male':
                        new_text = self.mask_byDict(text, self.terms_f2m)
                        idis.append(new_text)
                    elif identity == 'male' and identity_idi == 'female':
                        new_text = self.mask_byDict(text, self.terms_m2f)
                        idis.append(new_text)
                    else:
                        raise ValueError("text contains no gender related terms")
        return idis

    def generate_cda(self, text):
        idis = []
        identity_index = []
        for i in range(0, len(self.identities)):
            identity = self.identities[i]
            if self.identity_detect(text, identity) == True:
                identity_index.append(i)
        if len(identity_index) == 0:
            return False
        else:
            identity_idi_considered = []
            for index in identity_index:
                identity = self.identities[index]
                identity_idis = self.identity_idi_map[identity]
                identity_idis = [i for i in identity_idis if i not in identity_idi_considered]
                identity_idi_considered = identity_idi_considered + identity_idis
                for identity_idi in identity_idis:
                    if identity == 'female' and identity_idi == 'male':
                        print("-->text", text)
                        print(self.terms_f2m)
                        new_text = self.mask_byDict(text, self.terms_f2m)
                        print("-->new_text", new_text)
                        idis.append(new_text)
                    elif identity == 'male' and identity_idi == 'female':
                        new_text = self.mask_byDict(text, self.terms_m2f)
                        idis.append(new_text)
                    else:
                        # print("-->identity", identity)
                        # print("-->identity_idi", identity_idi)
                        new_text = self.replace_identity(text, self.identity_terms_map[identity], identity_idi)
                        idis.append(new_text)
        return idis




# orig_list = []
# male_list = []
# female_list = []
# homosexual_list = []
# christian_list = []
# muslim_list = []
# jewish_list = []
# black_list = []
# white_list = []
# illness_list = []
# label_list = []
# all_identity = [orig_list, male_list, female_list, homosexual_list, christian_list, muslim_list, jewish_list, black_list,
#                 white_list, illness_list, label_list]
#
# dataset = pd.read_csv("dataset/IMDB_train.csv")
# texts = dataset['text']
# labels = dataset['label']
# for j in range(0, len(texts)):
#     text = texts[j]
#     if identity_detect(text, all_identity_terms) == True:
#         all_identity[0].append(text)
#         all_identity[-1].append(labels[j])
#         for i in range(0, len(identities)):
#             identity = identities[i]
#             replace_text = replace_identity(text, all_identity_terms, identity)
#             all_identity[i + 1].append(replace_text)
#             # if identity_detect(text, identity) == True:
#             #     all_identity[i+1].append(text)
#             # else:
#             #     replace_text = replace_identity(text, all_identity_terms, identity)
#             #     all_identity[i+1].append(replace_text)
#
# dict = {"orig_text": orig_list, "male": male_list, "female": female_list, "homosexual": homosexual_list,
#         "christian": christian_list, "muslim": muslim_list, "jewish": jewish_list, "black": black_list,
#         "white": white_list, "psychiatric_or_mental_illness": illness_list, "label": label_list}
# dataset_identity = DataFrame(dict)
# print("-->dataset_identity", dataset_identity)
# dataset_identity.to_csv("dataset/IDM_train_identity.csv")

def main():
    ID = IdentityDetect()
    ##### generate cda data
    # text = "it 's funny , she even taught their dogs to hate blacks ."
    # idis = ID.generate_idi(text)
    # print("-->idis", idis)

    # ID.all_identity_terms = ID.term_class_dict["male"] + ID.term_class_dict["female"]
    #
    # text_list = []
    # label_list = []
    # dataset = pd.read_csv("dataset/hate_speech_online/reddit/train.csv")
    # print("-->dataset", dataset)
    # texts = dataset['text'].tolist()
    # labels = dataset['label'].tolist()
    # # text_list = texts.copy()
    # # label_list = labels.copy()
    # for j in tqdm(range(0, len(texts))):
    #     text = texts[j]
    #     try:
    #         text = text.lower()
    #     except:
    #         print("-->text", text)
    #         continue
    #     if ID.identity_detect(text, ID.all_identity_terms) == True:
    #         # all_identity[0].append(text)
    #         # text_list.append(text)
    #         # all_identity[-1].append(labels[j])
    #         # label_list.append(labels[j])
    #         idis = ID.generate_gender_version(text)
    #         # if idis == False:
    #         #     continue
    #         for idi in idis:
    #             text_list.append(idi)
    #             label_list.append(labels[j])
    #
    # dict = {"text": texts + text_list, "label": labels + label_list}
    # # dict = {"text": texts + text_list, "label": labels + label_list}
    # dataset_identity = DataFrame(dict)
    # print("-->dataset_identity", dataset_identity)
    # dataset_identity.to_csv("dataset/hate_speech_online/reddit/train_cda_gender.csv")
    #
    # text_list = []
    # label_list = []
    # dataset = pd.read_csv("dataset/hate_speech_online/reddit/test.csv")
    # print("-->dataset", dataset)
    # texts = dataset['text'].tolist()
    # labels = dataset['label'].tolist()
    # # text_list = texts.copy()
    # # label_list = labels.copy()
    # for j in tqdm(range(0, len(texts))):
    #     text = texts[j]
    #     try:
    #         text = text.lower()
    #     except:
    #         print("-->text", text)
    #         continue
    #     if ID.identity_detect(text, ID.all_identity_terms) == True:
    #         # all_identity[0].append(text)
    #         # text_list.append(text)
    #         # all_identity[-1].append(labels[j])
    #         # label_list.append(labels[j])
    #         idis = ID.generate_gender_version(text)
    #         # if idis == False:
    #         #     continue
    #         for idi in idis:
    #             text_list.append(idi)
    #             label_list.append(labels[j])
    #
    # dict = {"text": texts + text_list, "label": labels + label_list}
    # dataset_identity = DataFrame(dict)
    # print("-->dataset_identity", dataset_identity)
    # dataset_identity.to_csv("dataset/hate_speech_online/reddit/test_cda_gender.csv")



    # text = " mother is nurse and used to be a girl and father is a doctor. "
    # print(ID.generate_gender_version(text))
    # print(ID.generate_idi(text))
    # print(ID.replace_identity(text, ID.gender_terms, "male"))

    # text = "i am a mother and also a jewish"
    # print(ID.generate_idi(text))
    # return

    ##### generate idis
    # dataset_train = pd.read_csv("dataset/IMDB_train_identity.csv")
    # dataset_test = pd.read_csv("dataset/IMDB_test_identity.csv")
    # texts = dataset_train['orig_text'].tolist()
    # labels = dataset_train['label'].tolist()
    # all_texts = []
    # all_labels = []
    # for j in range(0, len(ID.identities)):
    #     identity = ID.identities[j]
    #     print("-->identity", identity)
    #     texts_identity = []
    #     label_identity = []
    #     for i in range(0, len(texts)):
    #         text = texts[i]
    #         if ID.identity_detect(text, identity) == True:
    #             texts_identity.append(text)
    #             label_identity.append(labels[i])
    #     all_texts.append(texts_identity)
    #     all_labels.append(label_identity)

    # for j in range(0, len(ID.identities)):
    #     identity = ID.identities[j]
    #     print("-->identity", identity)
    #     datas = {"text": all_texts[j], "label": all_labels[j]}
    #     dataset = DataFrame(datas)
    #     print("-->dataset", dataset)
    #     dataset.to_csv("dataset/IMDB_train" + identity + ".csv")

    # train_data = pd.read_csv("dataset/IMDB_train.csv")
    # test_data = pd.read_csv("dataset/IMDB_test.csv")
    # # all_data = train_data.append(test_data)
    #
    # print("-->train_data", train_data)
    # print("-->test_data", test_data)
    #
    # train_identity = pd.read_csv("dataset/IMDB_train_identity.csv")
    # test_identity = pd.read_csv("dataset/IMDB_test_identity.csv")
    #
    # print("-->train_identity", train_identity)
    # print("-->test_identity", test_identity)

    # train_df, test_df = train_test_split(all_data, test_size=0.3, random_state=999)
    #
    # print("-->train_df", train_df)
    # print("-->test_df", test_df)
    #
    # train_df.to_csv("dataset/IMDB/train_random.csv")
    # test_df.to_csv("dataset/IMDB/test_random.csv")

    # dataset = pd.read_csv("dataset/wiki_train.csv")
    # labels = dataset['label'].tolist()  # 1:toxic, 0: not toxic
    # sum = 0
    # for l in labels:
    #     if l == 1:
    #         sum += 1
    # print("percentage of toxic label", sum/float(len(labels)))
    #
    # dataset = pd.read_csv("dataset/wiki_train_identity.csv")
    # dataset = dataset[:5]
    # print("-->dataset", dataset)
    # print(dataset['orig_text'].tolist())
    # print(dataset['idis'].tolist())
    # print(dataset['label'].tolist())

    """
    orig_list = []
    idis_list = []
    label_list = []
    all_identity = [orig_list, idis_list, label_list]
    dataset = pd.read_csv("dataset/hate_speech_offensive/train.csv")
    print("-->dataset", dataset)
    texts = dataset['text']
    labels = dataset['label']
    for j in tqdm(range(0, len(texts))):
        text = texts[j]
        try:
            text = text.lower()
        except:
            print("-->text", text)
            continue
        if ID.identity_detect(text, ID.all_identity_terms) == True:
            all_identity[0].append(text)
            all_identity[-1].append(labels[j])
            idis = ID.generate_idi(text)
            idis_list.append(idis)

    dict = {"orig_text": orig_list, "idis": idis_list, "label": label_list}
    dataset_identity = DataFrame(dict)
    print("-->dataset_identity", dataset_identity)
    dataset_identity.to_csv("dataset/hate_speech_offensive/train_identity.csv")

    orig_list = []
    idis_list = []
    label_list = []
    all_identity = [orig_list, idis_list, label_list]
    dataset = pd.read_csv("dataset/hate_speech_offensive/test.csv")
    texts = dataset['text']
    labels = dataset['label']
    for j in tqdm(range(0, len(texts))):
        text = texts[j]
        try:
            text = text.lower()
        except:
            print("-->text", text)
            continue
        if ID.identity_detect(text, ID.all_identity_terms) == True:
            orig_list.append(text)
            label_list.append(labels[j])
            idis = ID.generate_idi(text)
            idis_list.append(idis)

    dict = {"orig_text": orig_list, "idis": idis_list, "label": label_list}
    dataset_identity = DataFrame(dict)
    print("-->dataset_identity", dataset_identity)
    dataset_identity.to_csv("dataset/hate_speech_offensive/test_identity.csv")
    """

    orig_list = []
    idis_list = []
    label_list = []
    def generated_cd_data(dataset_name, save_dataset_name):
        ID.all_identity_terms = ID.term_class_dict["male"] + ID.term_class_dict["female"]
        # training dataset
        path = "dataset/" + dataset_name + "/train.csv"
        dataset = pd.read_csv(path)
        print("-->dataset", dataset)
        texts = dataset['text']
        labels = dataset['label']
        new_texts = []
        new_labels = []
        for j in tqdm(range(0, len(texts))):
            text = texts[j]
            label = labels[j]
            try:
                text = text.lower()
            except:
                print("-->text", text)
                continue
            new_texts.append(text)
            new_labels.append(label)
            if ID.identity_detect(text, ID.all_identity_terms) == True:
                idis = ID.generate_gender_version(text)  # generate_idi
                for idi in idis:
                    new_texts.append(idi)
                    new_labels.append(label)

        # testing dataset
        dict = {"text": new_texts, "label": new_labels}
        dataset_identity = DataFrame(dict)
        print("-->dataset_identity", dataset_identity)
        save_path = "dataset/" + save_dataset_name + "/train_cda_gender.csv"
        dataset_identity.to_csv(save_path)

    # print("hate_speech_white")
    # generated_cd_data("hate_speech_white", "hate_speech_white")
    # print("hate_speech_twitter")
    # generated_cd_data("hate_speech_twitter", "hate_speech_twitter")
    # print("hate_speech_offensive")
    # generated_cd_data("hate_speech_offensive", "hate_speech_offensive")
    print("hate_speech_online_gab")
    generated_cd_data("hate_speech_online/gab", "hate_speech_online/gab")
    print("hate_speech_online_reddit")
    generated_cd_data("hate_speech_online/reddit", "hate_speech_online/reddit")


    # dataset = pd.read_csv("dataset/hate_speech_offensive/test.csv")
    # print("-->dataset", dataset)
    # texts = dataset['text']
    # labels = dataset['label']
    # new_texts = []
    # new_labels = []
    # for j in tqdm(range(0, len(texts))):
    #     text = texts[j]
    #     label = labels[j]
    #     try:
    #         text = text.lower()
    #     except:
    #         print("-->text", text)
    #         continue
    #     new_texts.append(text)
    #     new_labels.append(label)
    #     if ID.identity_detect(text, ID.all_identity_terms) == True:
    #         idis = ID.generate_gender_version(text)
    #         for idi in idis:
    #             new_texts.append(idi)
    #             new_labels.append(label)
    #
    # dict = {"text": new_texts, "label": new_labels}
    # dataset_identity = DataFrame(dict)
    # print("-->dataset_identity", dataset_identity)
    # dataset_identity.to_csv("dataset/hate_speech_offensive/test_cda_gender.csv")

if __name__ == '__main__':
    main()

# dataset_orig = pd.read_csv("dataset/dataset_identity1.csv")
# dataset_after = pd.read_csv("dataset/dataset_identity.csv")
#
# text_orig = dataset_orig['male']
# text_after = dataset_after['male']

# for identity in identities:
#     text_identity = dataset_orig[identity]
#     for i in range(0, len(text_identity)):
#         # extract identity words?





# dataset_dev = pd.read_csv("dataset/wiki_dev_orig.csv")
# dataset_test = pd.read_csv("dataset/wiki_test_orig.csv")
# dataset_train = pd.read_csv("dataset/wiki_train_orig.csv")
# frames = [dataset_dev, dataset_test, dataset_train]
# dataset_all = pd.concat(frames)
# dataset_all.to_csv("dataset/wiki_all.csv")


# dataset_all = pd.read_csv("dataset/wiki_all.csv")
#
# dataset_train = dataset_all.sample(frac=0.7, random_state=999)
# dataset_test = dataset_all.drop(dataset_train.index)
# print("-->dataset_train", dataset_train)
# print("-->dataset_test", dataset_test)
#
# dataset_train.to_csv("dataset/wiki_train1.csv")
# dataset_test.to_csv("dataset/wiki_test1.csv")
#
# dataset_train = pd.read_csv("dataset/wiki_train1.csv")
# dataset_test = pd.read_csv("dataset/wiki_test1.csv")
#
# print("-->dataset_train", dataset_train)
# print("-->dataset_test", dataset_test)
#
# text = []
# label = []
# for i in range(0, len(dataset_train)):
#     text.append(dataset_train['comment'][i])
#     label.append(dataset_train['is_toxic'][i])
#
# label_num = [1 if l == True else 0 for l in label]
# dict = {"text": text, "label": label_num}
# dataset = DataFrame(dict)
# dataset.to_csv("dataset/wiki_train.csv")
#
# text = []
# label = []
# for i in range(0, len(dataset_test)):
#     text.append(dataset_test['comment'][i])
#     label.append(dataset_test['is_toxic'][i])
#
# label_num = [1 if l == True else 0 for l in label]
# dict = {"text": text, "label": label_num}
# dataset = DataFrame(dict)
# dataset.to_csv("dataset/wiki_test.csv")
#
# dataset_train = pd.read_csv("dataset/wiki_train.csv")
# dataset_test = pd.read_csv("dataset/wiki_test.csv")
#
# print("-->dataset_train", dataset_train)
# print("-->dataset_test", dataset_test)





# dataset_identity = pd.read_csv("dataset/dataset_identity1.csv")
# # dataset_identity.sample(frac=1, random_state=999)
#
# dataset_identity = dataset_identity.sample(frac=0.7, random_state=999)
# print("-->dataset_identity", dataset_identity)
#
#
# print("-->keys", dataset_identity.keys())
#
# print("-->labels", dataset_identity['label'].tolist().count(1))
# print('-->length', len(dataset_identity))
#
#
# def get_metrics_single(self, model, dataset_identity, tokenizer, padding, max_seq_length, metric):
#     all_labels = []
#     for i in range(len(identities)):
#         identity = identities[i]
#         print("-->identity", identity)
#
#         datas = {'text': dataset_identity[identity], 'label': dataset_identity['label']}
#         dataset = DataFrame(datas)
#
#         logits, labels = self.get_predictions_single(model=model, dataset=dataset, tokenizer=tokenizer,
#                                                      padding=padding,
#                                                      max_seq_length=max_seq_length, if_identity=False)
#         all_labels.append(labels)
#
#     sum = np.array(all_labels[0]) + np.array(all_labels[1]) + np.array(all_labels[2]) + np.array(all_labels[3]) + \
#           np.array(all_labels[4]) + np.array(all_labels[5]) + np.array(all_labels[6]) + np.array(
#         all_labels[7]) + np.array(all_labels[8])
#     sum = sum.tolist()
#     sum_all = 0
#     for j in sum:
#         if j / 9.0 != 0 and j / 9.0 != 1:
#             sum_all += 1
#     ns = sum_all
#     print("-->ns", sum_all)
#     n = len(all_labels[0])
#     individual_metric = ns / float(n)
#     print("-->individual metric:", individual_metric)
#     return individual_metric
#
#
# model = torch.load('added_model_mse.pkl')
# metric = self.get_metrics_single(model, dataset_identity, tokenizer, padding, max_seq_length, 'fnr')





