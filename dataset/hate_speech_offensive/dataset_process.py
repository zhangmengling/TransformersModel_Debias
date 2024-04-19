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

seed = 999
random.seed(seed)


texts = []
annotations = []

with open("dataset.json", "r") as f:
    for line in f:
        try:
           datas = json.loads(line)
        except:
            print("None")
            continue


# {"1179055004553900032_twitter": {"post_id": "1179055004553900032_twitter",
#                                  "annotators": [{"label": "normal",
#                                                  "annotator_id": 1,
#                                                  "target": ["None"]},
#                                                 {"label": "normal",
#                                                  "annotator_id": 2,
#                                                  "target": ["None"]},
#                                                 {"label": "normal",
#                                                  "annotator_id": 3,
#                                                  "target": ["None"]}],
#                                  "rationales": [],
#                                  "post_tokens": ["i", "dont", "think", "im", "getting", "my", "baby", "them", "white", "9", "he", "has", "two", "white", "j", "and", "nikes", "not", "even", "touched"]}
# for data in datas:

texts = []
labels = []

for key, value in datas.items():
    data = value
    tokens = data['post_tokens']
    text = ' '.join(tokens)
    texts.append(text)
    annotators = data['annotators']
    labels_one = []
    for i in range(0, len(annotators)):
        annotate = annotators[i]
        label = annotate['label']
        if label == 'normal':
            labels_one.append(0)
        else:
            labels_one.append(1)
    if sum(labels_one) / float(len(labels_one)) <= 0.5:
        labels.append(0)
    else:
        labels.append(1)

print(len(texts))
print(len(labels))
datas = {"text": texts, "label": labels}
dataset = pd.DataFrame(datas)
# dataset.to_csv("all_data.csv")

train_df, test_df = train_test_split(dataset, test_size=0.3, random_state=999)
print("-->train_df", train_df)
print("-->test_df", test_df)
train_df.to_csv("train.csv")
test_df.to_csv("test.csv")



