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

with open("amateur_expert.json", "r") as f:
    for line in f:
        try:
           data = json.loads(line)
        except:
            print("None")
            continue
        # data = dict(line)
        text = data['text']
        texts.append(text)

        annotation = data['Annotation']
        annotations.append(annotation)
    # data = json.load(f)

with open("neither.json", "r") as f:
    for line in f:
        try:
           data = json.loads(line)
        except:
            print("None")
            continue
        # data = dict(line)
        text = data['text']
        texts.append(text)

        annotation = data['Annotation']
        annotations.append(annotation)

with open("racism.json", "r") as f:
    for line in f:
        try:
           data = json.loads(line)
        except:
            print("None")
            continue
        # data = dict(line)
        text = data['text']
        texts.append(text)

        annotation = data['Annotation']
        annotations.append(annotation)

with open("sexism.json", "r") as f:
    for line in f:
        try:
           data = json.loads(line)
        except:
            print("None")
            continue
        # data = dict(line)
        text = data['text']
        texts.append(text)

        annotation = data['Annotation']
        annotations.append(annotation)

labels = []
for a in annotations:
    if a == "Neither" or a =="none" or a == "None":
       labels.append(0)
    else:
        labels.append(1)

unique_texts = []
unique_labels = []
for i in range(0, len(texts)):
    t = texts[i]
    if t not in unique_texts:
        unique_texts.append(t)
        unique_labels.append(labels[i])

print(len(texts))
print(len(labels))
datas = {"text": texts, "label": labels}
dataset = pd.DataFrame(datas)
dataset = dataset.sample(frac=1, random_state=999)
print("-->dataset", dataset)
dataset.to_csv("all_data.csv")

train_df, test_df = train_test_split(dataset, test_size=0.3, random_state=999)
print("-->train_df", train_df)
print("-->test_df", test_df)
train_df.to_csv("train.csv")
test_df.to_csv("test.csv")
