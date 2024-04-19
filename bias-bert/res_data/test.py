import sys, os
sys.path.append('./..')
# from train_functions import check_path
import pandas as pd
import data_masking as masking
import re
import pandas as pd

# file_path = './IMDB_l_train'
# file_path = './IMDB_gender_train.csv'
# train_file = pd.read_csv(file_path)
# file_path = './IMDB_gender_test.csv'
# test_file = pd.read_csv(file_path)
#
# train_file_ = train_file.copy()
# indexs = []
# for i in range(0, len(train_file_['text'])):
#     count_total = train_file_['count_total'][i]
#     if count_total == 0:
#         indexs.append(i)
#
# train_file.drop(index=indexs)
# train_file.to_csv('IMDB_gender_train.csv')
#
# test_file_ = test_file.copy()
# indexs = []
# for i in range(0, len(test_file_['text'])):
#     count_total = test_file_['count_total'][i]
#     if count_total == 0:
#         indexs.append(i)
#
# test_file.drop(index=indexs)
# test_file.to_csv('IMDB_gender_test.csv')
#
# print("-->train_file", train_file)
# print("-->test_file", test_file)

# print("-->file_path", file_path)
# unpickled_df = pd.read_pickle(file_path)
# print("-->", unpickled_df)
# print(unpickled_df.keys())
#
# print(unpickled_df['text'][0])
# print(unpickled_df['text_all_M'][0])
# print(unpickled_df['text_all_F'][0])
#
# print("-->number of dataset", unpickled_df.shape[0])
# unpickled_df.to_csv("IMDB_gender_train.csv")


df_train = pd.read_csv("dataset_IMDB.csv")
df_test = pd.read_csv("dataset_test_IMDB.csv")

##### ##### #####
# IMDb - Step 1: Gender neutral data sets for training
df_train_ = df_train.copy()
df_test_ = df_test.copy()

print("-->df_train_", df_train_)
print(df_train_.keys())

masking.make_all_df(df_train_)
masking.make_all_df(df_test_)

masking.check_df(df_test_)
masking.check_df(df_train_)

# Safe whole table (large)
df_train_.to_csv("IMDB_gender_train.csv")
df_test_.to_csv("IMDB_gender_test.csv")
print("-->df_train_", df_train_)
print("-->df_test_", df_test_)

# df_train_.to_pickle(path + "IMDB_l_train.csv")
# df_test_.to_pickle(path + "IMDB_l_test")

# equal_index = []
# for i in range(0, unpickled_df.shape[0]):
#     if len(unpickled_df['text_all_M'][i]) <= 50:
#         print("-->M:", unpickled_df['text_all_M'][i])
#         print("-->F:", unpickled_df['text_all_F'][i])
#     if unpickled_df['text_all_M'][i] == unpickled_df['text_all_F'][i]:
#         equal_index.append(i)
#
# print("-->equal_index", len(equal_index))
# df = unpickled_df.drop(index=equal_index)
# df.to_pickle("./IMDB_gender_train")

#
# for spec in ['_all', '_pro', '_weat']:
#     file_path = 'IMDB_training/IMDB_MIN_mix' + spec + '_test'
#     print("-->file_path", file_path)
#     unpickled_df = pd.read_pickle(file_path)
#     print("-->", unpickled_df)
#     print(unpickled_df.keys())
#
# texts = ["This flick is a waste of time.I expect from an action movie to have more than 2 explosions and some shooting.Van Damme's acting is awful. He never was much of an actor, but here it is worse.He was definitely better in his earlier movies. His screenplay part for the whole movie was probably not more than one page of stupid nonsense one liners.The whole dialog in the film is a disaster, same as the plot.The title \"The Shepherd\" makes no sense. Why didn't they just call it \"Border patrol\"? The fighting scenes could have been better, but either they weren't able to afford it, or the fighting choreographer was suffering from lack of ideas.This is a cheap low type of action cinema."]
#
# # Standard imports
# import logging, re, pickle, os, nltk, random #, en_core_web_sm, spacy
# logging.basicConfig(level=logging.INFO)
# from term_lists import *
#
# logging.info("successfully imported the latest version of data_masking.")
# print("successfully imported the latest version of data_masking.")
# # ----------------------------------------------------------------------------------------------- #
# # ----------------------------------------------------------------------------------------------- #
# # utility functions
# def add_space(word):
#     return ' ' + word + ' '
#
# # def add_space_behind(word):
# #     return word + ' '
#
# # ----------------------------------------------------------------------------------------------- #
# def count_terms(text, terms=all_terms):
#     res = dict.fromkeys(terms, 0)
#     for elem in terms:
#         res[elem] = text.count(add_space(elem))
#     return res
#
# characters = [',', ':', '.', '?', '!', '`', '@', '#', '$', '%', '^', '&', '*', '/', ';', '[', ']', '{', '}', '(', ')']
#
#
# def consider_characters(term):
#     all_terms = []
#     for cha in characters:
#         all_terms.append(' ' + term + cha)
#         all_terms.append(cha + term + ' ')
#     return all_terms
#
#
# def mask_byDict(review, terms):
#     '''
#     mask_byDict: Mask terms in a text
#     args
#         review (str): Text
#         terms (dict): tems. Mask kes by value.
#     return tuple [(str) new masked review text, (dict) term occurances]
#     '''
#
#     count_dict = {}
#     review = review.lower()
#     for word, initial in terms.items():
#         # count_dict[word] = len(re.findall(add_space(word), review))
#         review = review.replace(add_space(word), add_space(initial))
#
#         extending_word = consider_characters(word)
#         extending_initial = consider_characters(initial)
#         for i in range(0, len(extending_word)):
#             review = review.replace(extending_word[i], extending_initial[i])
#     return review
#
# def make_male(review):
#     return mask_byDict(review, terms_f2m)
#
# def make_female(review):
#     return mask_byDict(review, terms_m2f)
#
# def make_neutral(text, terms=all_terms):
#     for elem in terms:
#         text = text.replace(add_space(elem), ' ')
#     return text
#
# text_all_M = [mask_byDict(e, terms_f2m) for e in texts]
# print("-->text_all_M", text_all_M)
# text_all_F = [mask_byDict(e, terms_m2f) for e in texts]
# print("-->text_all_F", text_all_F)
#
