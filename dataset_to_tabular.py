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

from fair_metrics import FairMetrics
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from multiprocessing.dummy import Pool as ThreadPool
from model_operation import SubModel
from custom_model_operation import CustomModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
import re
from pandas.core.frame import DataFrame

from sklearn.model_selection import train_test_split

from dataset_process import IdentityDetect

seed = 999
random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("-->device", device)
torch.set_grad_enabled(True)

# train_texts = dataset_train['text']
# train_labels = dataset_train['label']
# test_texts = dataset_test['text']
# test_labels = dataset_test['label']
# train_texts_identity = []
# train_labels_identity = []
# test_texts_identity = []
# test_labels_identity = []
# gender_attribute = []
# religion_attribute = []
# race_attribute = []

identity_to_attribute = {"male": "gender", "female": "gender", "homosexual": "gender",
                        "christian": "religion", "muslim": "religion", "jewish": "religion",
                        "black": "race", "white": "race"}
identity_to_num = {"male": 1, "female": 2, "homosexual": 3,
                        "christian": 1, "muslim": 2, "jewish": 3,
                        "black": 1, "white": 2}
def data2tabular(dataset):
    texts_id = []
    labels_identity = []
    gender_attribute = []
    religion_attribute = []
    race_attribute = []

    ID = IdentityDetect()
    identities = list(ID.term_class_dict.keys())
    print("-->identities", identities)

    texts = dataset['text']
    labels = dataset['label']
    for j in tqdm(range(0, len(texts))):
        gender_a = 0
        religion_a = 0
        race_a = 0
        text = texts[j]
        label = labels[j]
        try:
            text = text.lower()
        except:
            print("-->text", text)
            continue
        texts_id.append(j)
        labels_identity.append(label)
        if ID.identity_detect(text, ID.all_identity_terms) == True:
            contain_identity = ID.which_identity(text)
            for identity in contain_identity:
                attribute = identity_to_attribute[identity]
                if attribute == "gender":
                    gender_a = identity_to_num[identity]
                if attribute == "religion":
                    religion_a = identity_to_num[identity]
                if attribute == "race":
                    race_a = identity_to_num[identity]
        gender_attribute.append(gender_a)
        religion_attribute.append(religion_a)
        race_attribute.append(race_a)

    datas = {"text_id": texts_id, "gender": gender_attribute, "religion": religion_attribute,
             "race": race_attribute, "label": labels_identity}
    dataset = pd.DataFrame(datas)
    return dataset

base_path = "dataset/"
dataset_name = "hate_speech_twitter"
dataset_train = pd.read_csv(base_path + dataset_name + "/train.csv")
dataset_test = pd.read_csv(base_path + dataset_name + "/test.csv")

dataset = data2tabular(dataset_test)
print("-->dataset", dataset)
dataset.to_csv("dataset/hate_speech_online_twitter_test.csv")
