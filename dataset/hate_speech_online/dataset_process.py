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

dataset = pd.read_csv("data/reddit.csv")
texts_orig = dataset['text']
hate_speech_index = dataset['hate_speech_idx']

texts = []
labels = []

for i in range(0, len(texts_orig)):
    text = texts_orig[i]
    try:
        indexs = eval(hate_speech_index[i])
    except:
        continue

    # Split the text into conversations using regex
    conversations = re.split(r'\d+\.', text)

    # Remove leading/trailing white space from each conversation
    conversations = [conv.strip() for conv in conversations if conv.strip()]

    for j in range(len(conversations)):
        texts.append(conversations[j])
        if j+1 in indexs:
            labels.append(1)
        else:
            labels.append(0)

print("label's is 1: {}, all:{}".format(sum(labels), len(labels)))

datas = {"text": texts, "label": labels}
dataset = pd.DataFrame(datas)
dataset = dataset.sample(frac=1, random_state=999)
print("-->dataset", dataset)

train_df, test_df = train_test_split(dataset, test_size=0.3, random_state=999)
print("-->train_df", train_df)
print("-->test_df", test_df)
train_df.to_csv("train.csv")
test_df.to_csv("test.csv")