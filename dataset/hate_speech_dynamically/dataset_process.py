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

dataset = pd.read_csv("Dynamically Generated Hate Dataset v0.2.3.csv")
text = dataset['text']
label_orig = dataset['label']

label = []
for l in label_orig:
    if l == "hate":
        label.append(1)
    elif l == "nothate":
        label.append(0)
    else:
        print("Error")

datas = {"text":text, "label":label}
dataset = pd.DataFrame(datas)
print("-->dataset", dataset)
# dataset.to_csv()

train_df, test_df = train_test_split(dataset, test_size=0.3, random_state=999)
print("-->train_df", train_df)
print("-->test_df", test_df)
train_df.to_csv("train.csv")
test_df.to_csv("test.csv")

