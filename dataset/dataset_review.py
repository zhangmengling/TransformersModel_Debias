import random

from datasets import load_dataset, load_metric

import pandas as pd
import copy
import torch

from tqdm import tqdm
import numpy as np
import json
import re
from pandas.core.frame import DataFrame

from sklearn.model_selection import train_test_split

"""
WhiteForumHate
"""
# dataset = pd.read_csv("hate_speech_white/all_data.csv")
# print("all", dataset)
# labels = dataset['label'].tolist()
# print("-->lenght", len(labels))
# print("hate num", labels.count(1))

# dataset = pd.read_csv("hate_speech_twitter/all_data.csv")
# print("-->Dataset", dataset)
#
# texts = []
# annotations = []
# with open("hate_speech_twitter/amateur_expert.json", "r") as f:
#     for line in f:
#         try:
#            data = json.loads(line)
#         except:
#             print("None")
#             continue
#         # data = dict(line)
#         text = data['text']
#         texts.append(text)
#
#         annotation = data['Annotation']
#         annotations.append(annotation)
# print("-->amateur_expert:", len(texts))
# print(len(annotations))
# print("-->Sexism", annotations.count("Sexism"))
# print("-->Racism", annotations.count("Racism"))
# print("-->Both", annotations.count("Both"))
# print("-->Neither", annotations.count("Neither"))
#
# print(list(set(annotations)))
# non_hate_num = 0
# sexisum_num = 0
# racism_num = 0
# both_num = 0
# for a in annotations:
#     if a == "Neither" or a =="none" or a == "None":
#         non_hate_num += 1
#     elif a == "Sexism":
#         sexisum_num += 1
#     elif a == "Racism":
#         racism_num += 1
#     elif a == "Both":
#         both_num += 1
# print("-->non_hate_num", non_hate_num)
# print("-->sexisum_num", sexisum_num)
# print("-->racism_num", racism_num)
# print("-->both_num", both_num)
#
# texts = []
# with open("hate_speech_twitter/neither.json", "r") as f:
#     for line in f:
#         try:
#            data = json.loads(line)
#         except:
#             print("None")
#             continue
#         # data = dict(line)
#         text = data['text']
#         texts.append(text)
# print("-->neither:", len(texts))
#
# texts = []
# with open("hate_speech_twitter/racism.json", "r") as f:
#     for line in f:
#         try:
#            data = json.loads(line)
#         except:
#             print("None")
#             continue
#         # data = dict(line)
#         text = data['text']
#         texts.append(text)
# print("-->racism:", len(texts))
#
# texts = []
# with open("hate_speech_twitter/sexism.json", "r") as f:
#     for line in f:
#         try:
#            data = json.loads(line)
#         except:
#             print("None")
#             continue
#         # data = dict(line)
#         text = data['text']
#         texts.append(text)
# print("-->sexism:", len(texts))
#
# dataset = pd.read_csv("hate_speech_twitter/train.csv")
# print("-->train Dataset", dataset)
#
# dataset = pd.read_csv("hate_speech_twitter/test.csv")
# print("-->test Dataset", dataset)
#


"""
hate speech from gab
"""

# dataset = pd.read_csv("hate_speech_online/reddit/train.csv")
# print("train", dataset)
# labels = dataset['label'].tolist()
# print("hate num", labels.count(1))
# dataset = pd.read_csv("hate_speech_online/reddit/test.csv")
# print("test", dataset)
# labels = dataset['label'].tolist()
# print("-->lenght", len(labels))
# print("hate num", labels.count(1))

# """
# hate and offensive speech from twitter and gab
# """
#
# dataset = pd.read_csv("hate_speech_offensive/train.csv")
# print("train", dataset)
# labels = dataset['label'].tolist()
# print("hate num", labels.count(1))
# dataset = pd.read_csv("hate_speech_offensive/test.csv")
# print("test", dataset)
# labels = dataset['label'].tolist()
# print("-->lenght", len(labels))
# print("hate num", labels.count(1))

# dataset_idi = pd.read_csv("hate_speech_offensive/train_idis.csv")
# print(dataset_idi)
# texts = eval(dataset_idi['text'].tolist()[0])
# labels = eval(dataset_idi['label'].tolist()[0])
# print(len(texts), len(labels))
#
# datas = {'text': texts, "label": labels}
# dataset = pd.DataFrame(datas)
# print(dataset)
# dataset.to_csv("hate_speech_offensive/train_idis.csv")

dataset_idi = pd.read_csv("hate_speech_white/train_idis.csv")
dataset_train = pd.read_csv("hate_speech_white/train.csv")
dataset_comebine = pd.concat([dataset_train, dataset_idi])
dataset = dataset_comebine[['text', 'label']]
dataset.to_csv("hate_speech_white/train_combine.csv")

dataset_idi = pd.read_csv("hate_speech_twitter/train_idis.csv")
dataset_train = pd.read_csv("hate_speech_twitter/train.csv")
dataset_comebine = pd.concat([dataset_train, dataset_idi])
dataset = dataset_comebine[['text', 'label']]
dataset.to_csv("hate_speech_twitter/train_combine.csv")

dataset_idi = pd.read_csv("hate_speech_online/gab/train_idis.csv")
dataset_train = pd.read_csv("hate_speech_online/gab/train.csv")
dataset_comebine = pd.concat([dataset_train, dataset_idi])
dataset = dataset_comebine[['text', 'label']]
dataset.to_csv("hate_speech_online/gab/train_combine.csv")

dataset_idi = pd.read_csv("hate_speech_online/reddit/train_idis.csv")
dataset_train = pd.read_csv("hate_speech_online/reddit/train.csv")
dataset_comebine = pd.concat([dataset_train, dataset_idi])
dataset = dataset_comebine[['text', 'label']]
dataset.to_csv("hate_speech_online/reddit/train_combine.csv")

dataset_idi = pd.read_csv("hate_speech_offensive/train_idis.csv")
dataset_train = pd.read_csv("hate_speech_offensive/train.csv")
dataset_comebine = pd.concat([dataset_train, dataset_idi])
dataset = dataset_comebine[['text', 'label']]
dataset.to_csv("hate_speech_offensive/train_combine.csv")