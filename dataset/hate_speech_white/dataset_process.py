import random

from sklearn.metrics import davies_bouldin_score
from datasets import load_dataset, load_metric

import pandas as pd

from tqdm import tqdm
import numpy as np
import json
import re
from pandas.core.frame import DataFrame
import os

from sklearn.model_selection import train_test_split

seed = 999
random.seed(seed)


# def count_files_in_directory(directory):
#     return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
#
# train_dir = 'all_files/'
# num_files = count_files_in_directory(train_dir)
# print(f"There are {num_files} files in the directory.")
#
# test_dir = 'sampled_test/'
# num_files = count_files_in_directory(test_dir)
# print(f"There are {num_files} files in the directory.")


data_dir = 'all_files/'

annotation = pd.read_csv('annotations_metadata.csv')

text_ids = annotation['file_id']
labels_id = annotation['label']
print("-->labels_id", labels_id)
texts = []
labels = []

for i in range(0, len(text_ids)):
    label = labels_id[i]
    if label == 'noHate':
        labels.append(1)
    elif label == 'hate':
        labels.append(0)
    else:
        print(label)
        continue
    id = text_ids[i]
    try:
        file_name = data_dir + str(id) + ".txt"
        with open(file_name, 'r') as f:
            text = f.read()
        texts.append(text)
    except:
        print("no such file")

datas = {'text': texts, 'label': labels}
dataset = pd.DataFrame(datas)
print("-->dataset", dataset)
dataset.to_csv("./all_data.csv")

train_df, test_df = train_test_split(dataset, test_size=0.3, random_state=999)

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

train_df.to_csv("./train.csv")
test_df.to_csv("./test.csv")

print("-->train_df", train_df)
print("-->test_df", test_df)

# train_dataset = pd.read_csv('train.csv')
# train_dataset = train_dataset.drop(train_dataset.columns[0], axis=1)
# train_dataset.to_csv('train.csv')
#
# test_dataset = pd.read_csv('test.csv')
# test_dataset = test_dataset.drop(train_dataset.columns[0], axis=1)
# test_dataset.to_csv('test.csv')





