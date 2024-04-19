import os
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

# from fair_metrics import FairMetrics
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from multiprocessing.dummy import Pool as ThreadPool
# from model_operation import SubModel
# from custom_model_operation import CustomModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json
import re
from sklearn.metrics import confusion_matrix
from pandas.core.frame import DataFrame

import numpy as np
from sklearn.metrics import roc_auc_score
import traceback
import math

from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler

from .dataset_process import IdentityDetect
from .debias_editor import AddedLayers


import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

random.seed(999)
np.random.seed(999)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(999)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(True)

term_file_path = "public_operation/identity_class.json"
with open(term_file_path, "r") as term_file:
    term_class_dict = json.load(term_file)
identities = list(term_class_dict.keys())

characters = [',', ':', '.', '?', '!', '`', '@', '#', '$', '%', '^', '&', '*', '/', ';', '[', ']', '{', '}', '(', ')']

def consider_characters(term):
    all_terms = [term]
    for cha in characters:
        all_terms.append(' ' + term + cha)
        all_terms.append(cha + term + ' ')
        all_terms.append("'" + term + "'")
    return all_terms

def identity_detect(text, term_class):
    # text = text.lower().split(" ")
    text = re.split("[ ;.,:?!'/()\[\]*&`@#%$^{}]", text)
    # print("-->text", text)
    terms = term_class_dict[term_class]
    # print("-->terms", terms)
    for term in terms:
        extending_terms = consider_characters(term)
        for t in extending_terms:
            if t in text:
                return True
    return False

def identity_detect_with_term(text, term_class):
    text = re.split("[ ;.,:?!'/()\[\]*&`@#%$^{}]", text)
    terms = term_class_dict[term_class]
    exist_terms = []
    for term in terms:
        extending_terms = consider_characters(term)
        for t in extending_terms:
            if t in text:
                exist_terms.append(t)
    if len(exist_terms) > 0:
        return True, list(set(exist_terms))
    else:
        return False, []

# num_epochs = 10  # 60
# learning_rate = 0.001

class Optimization:
    def __init__(self, CustomModel, num_epochs, batch_size, learning_rate, target, privileged_label, threshold, gamma, train_data_name, test_data_name,
                 train_data_identity, test_data_identity, metric=None, optimizer=None):
        self.CustomModel = CustomModel
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.metric = metric
        self.optimizer = optimizer
        self.custom_module = self.CustomModel.custom_model["custom_module"]
        self.target = target
        self.privileged_label = privileged_label
        self.threshold = threshold
        self.gamma = gamma
        self.train_data_name = train_data_name
        self.train_data = pd.read_csv(train_data_name)
        self.test_data = pd.read_csv(test_data_name)
        self.train_data_identity = pd.read_csv(train_data_identity)
        self.test_data_identity = pd.read_csv(test_data_identity)

        self.BETA = 0.05
        self.ALPHA = 0.05
        self.sigma = 0.05
        self.MIN_BATCH_SIZE = 16
        self.MAX_BATCH_SIZE = 256
        self.STEP = 1

        # if self.target == "individual":
        #     self.threshold = 0.05
        # elif self.target == "group":
        #     self.threshold = 0.97
        self.orig_threshold = self.threshold

        self.unpri_label = 1
        self.pri_label = 0

    def metric_calculation(self, debias_dataset, tokenizer, padding, max_seq_length):
        text_all_M = debias_dataset['text_all_M'].values.tolist()[:10]
        text_all_F = debias_dataset['text_all_F'].values.tolist()[:10]
        diff = []
        for i in range(0, len(text_all_M)):
            # for the Male sentence
            text_M = text_all_M[i]
            # for the Female sentence
            text_F = text_all_F[i]
            inputs_M = tokenizer(text_M, padding=padding, max_length=max_seq_length, truncation=True,
                                 return_tensors="pt")
            # inputs_M.to(device)
            class_output_M, label_output_M = self.CustomModel.forward_with_custom_module(inputs_M,
                                                                                         self.CustomModel.custom_model[
                                                                                             "custom_module"])
            inputs_F = tokenizer(text_F, padding=padding, max_length=max_seq_length, truncation=True,
                                 return_tensors="pt")
            # inputs_F.to(device)
            class_output_F, label_output_F = self.CustomModel.forward_with_custom_module(inputs_F,
                                                                                         self.CustomModel.custom_model[
                                                                                             "custom_module"])
            # print("-->class_output_M:{}, class_output_F:{}".format(class_output_M, class_output_F))
            diff.append(torch.abs(torch.tensor(class_output_M[0][1]) - torch.tensor(class_output_F[0][1])))
        metric_score = torch.mean(torch.tensor(diff))
        metric_score.grad = None
        return metric_score

    def get_single_prediction(self, model, x_M, x_F):
        # x_M.to(device)
        # class_output_M, label_output_M = self.CustomModel.forward_with_custom_module(x_M,
        #                                                                              self.CustomModel.custom_model[
        #                                                                                  "custom_module"])
        base_M = self.CustomModel.get_base_bertmodel_output(x_M)
        # print("-->base_M", base_M)
        output_M = model.forward(base_M)
        # print("-->output_M", output_M)
        logits = self.CustomModel.custom_model['classifier'](output_M)
        # probability_M = self.CustomModel.custom_model['classifier'](output_M).tolist()[0][1]
        probability_M = self.CustomModel.custom_model['classifier'](output_M)[0][1]
        print("-->probability_M", probability_M)
        # x_F.to(device)
        # class_output_F, label_output_F = self.CustomModel.forward_with_custom_module(x_F,
        #                                                                              self.CustomModel.custom_model[
        #                                                                                  "custom_module"])
        base_F = self.CustomModel.get_base_bertmodel_output(x_F)
        output_F = model.forward(base_F)
        probability_F = self.CustomModel.custom_model['classifier'](output_F)[0][1]
        return probability_M, probability_F

        # diff_single = torch.abs(torch.tensor(class_output_M[0][1]) - torch.tensor(class_output_F[0][1]))
        # return torch.tensor(class_output_M[0][1]), torch.tensor(class_output_F[0][1])
        # return diff_single

    # def get_predictions(self, model, dataset, tokenizer, padding, max_seq_length, if_identity=False):
    #     """
    #     if model == None: no added linear function, use base_model to predict sampels
    #     if_identity = True: further output identity information
    #     """
    #     texts = dataset['text'].values.tolist()
    #     all_logits = []
    #     all_labels = []
    #     identities_result = []
    #     for text in tqdm(texts):
    #         input = tokenizer(text, padding=padding, max_length=max_seq_length, truncation=True,
    #                            return_tensors="pt")
    #         input.to(device)
    #
    #         if model == None:
    #             output = self.CustomModel.base_model(**input)
    #             logits = output.logits
    #             labels = [pro.index(max(pro)) for pro in logits.tolist()]
    #             all_logits.append(logits.tolist()[0])
    #             all_labels.append(labels[0])
    #         else:
    #             base = self.CustomModel.get_base_bertmodel_output(input)
    #             output = model.forward(base)
    #             logits = self.CustomModel.custom_model['classifier'](output)
    #             # print("-->logits", logits)
    #             labels = [pro.index(max(pro)) for pro in logits.tolist()]
    #             all_logits.append(logits.tolist()[0])
    #             all_labels.append(labels[0])
    #
    #         if if_identity == True:
    #             all_i = []
    #             for i in range(0, len(identities)):
    #                 identity = identities[i]
    #                 if identity_detect(text, identity) == True:
    #                     all_i.append(i)
    #                 if i == len(identities) - 1:
    #                     if all_i == []:
    #                         identities_result.append(-1)
    #                     elif len(all_i) == 1:
    #                         identities_result.append(all_i[0])
    #                     else:
    #                         identities_result.append(all_i)
    #                     all_i = []
    #
    #     if if_identity == True:
    #         return all_logits, all_labels, identities_result
    #     else:
    #         return all_logits, all_labels

    def get_predictions(self, model, dataset, tokenizer, padding, max_seq_length, if_identity=False):
        """
        predict all texts in dataset once as a whole brunch
        if model == None: no added linear function, use base_model to predict sampels
        if_identity = True: further output identity information
        """
        texts = dataset['text'].values.tolist()
        all_logits = []
        all_labels = []
        identities_result = []

        inputs = tokenizer(texts, padding=padding, max_length=max_seq_length, truncation=True,
                             return_tensors="pt")
        inputs.to(device)
        # print("-->inputs", inputs)
        base = self.CustomModel.get_base_bertmodel_output(inputs)
        # print("-->base", base)
        output = model.forward(base)
        # print("-->output", output)
        # logits_M = self.CustomModel.custom_model['classifier'](output_M)
        # print("-->logits_M", logits_M)
        logits = self.CustomModel.custom_model['classifier'](output)
        # print("-->logits", logits)
        labels = [pro.index(max(pro)) for pro in logits.tolist()]
        # print("-->probability_M", probability_M)
                        
        if if_identity == True:
            return logits, labels, identities_result
        else:
            return logits, labels

    def get_predictions_all_identity(self, model, classifier, dataset_identity, tokenizer, padding, max_seq_length, ID):
        """
        predict all texts in dataset once as a whole brunch
        if model == None: no added linear function, use base_model to predict sampels
        if_identity = True: further output identity information
        RETURN:
        all_logits: prediction logits for dataset_identity['orig_text'] as a whole
        all_labels: prediction labels for all texts
        overall_logits: prediction logits for texts batch of each identity in dataset_identity['orig_text']
        overall_labels: prediction labels for each subgroup, a list
        all_orig_labels: ground-truth labels for texts batch of each identity in dataset_identity['orig_text']
        """

        # ----------group metric on orig_text----------
        # datas = {'text': dataset_identity['orig_text'], 'label': dataset_identity['label']}
        # dataset = DataFrame(datas)
        texts = dataset_identity['orig_text'].tolist()
        labels = dataset_identity['label'].tolist()
        active = nn.Sigmoid()

        # # ----------group metric on all_texts----------
        # texts = []
        # labels = []
        # for i in range(len(dataset_identity['orig_text'])):
        #     orig_text = dataset_identity['orig_text'][i]
        #     idis = eval(dataset_identity['idis'][i])
        #     label = dataset_identity['label'][i]
        #     texts.append(orig_text)
        #     labels.append(label)
        #     for idi in idis:
        #         texts.append(idi)
        #         labels.append(label)

        # inputs = tokenizer(texts, padding=padding, max_length=max_seq_length, truncation=True,
        #                    return_tensors="pt")
        # inputs.to(device)
        # base = self.CustomModel.get_base_bertmodel_output(inputs)
        # if model != None:
        #     output = model.forward(base)
        # else:
        #     output = base64
        # if classifier != None:
        #     logits = classifier(output)
        # else:
        #     logits = self.CustomModel.custom_model['classifier'](output)
        # overall_logits = active(logits)
        # overall_labels = [pro.index(max(pro)) for pro in overall_logits.tolist()]

        all_texts = []
        all_base_labels = []
        for j in range(0, len(ID.identities)):
            identity = ID.identities[j]
            texts_identity = []
            label_identity = []
            for i in range(0, len(texts)):
                text = texts[i]
                if ID.identity_detect(text, identity) == True:
                    texts_identity.append(text)
                    label_identity.append(labels[i])
            all_texts.append(texts_identity)
            all_base_labels.append(label_identity)

        # prediction output for each subgroup withing one identity
        every_logits = []
        every_labels = []
        for i in range(0, len(all_texts)):
            identity = ID.identities[i]
            texts = all_texts[i]
            if len(texts) == 0:
                all_logits.append(None)
                all_labels.append(None)
                continue
            inputs = tokenizer(texts, padding=padding, max_length=max_seq_length, truncation=True,
                               return_tensors="pt")
            inputs.to(device)
            base = self.CustomModel.get_base_bertmodel_output(inputs)
            if model != None:
                output = model.forward(base)
            else:
                output = base64
            if classifier != None:
                logits = classifier(output)
            else:
                logits = self.CustomModel.custom_model['classifier'](output)
            logits = active(logits)
            labels = [pro.index(max(pro)) for pro in logits.tolist()]
            every_logits.append(logits)
            every_labels.append(labels)
        return every_logits, every_labels, all_base_labels
        # return every_logits, every_labels, overall_logits, overall_labels, all_base_labels

    def get_predictions_all_identity1(self, model, classifier, dataset_identity, tokenizer, padding, max_seq_length, ID):
        """
        predict all texts in dataset once as a whole brunch
        if model == None: no added linear function, use base_model to predict sampels
        if_identity = True: further output identity information
        RETURN:
        every_logits: prediction logits for dataset_identity['orig_text'] as a whole
        every_labels: prediction labels for all texts
        overall_logits: prediction logits for texts batch of each identity in dataset_identity['orig_text']
        overall_labels: prediction labels for each subgroup, a list
        all_orig_labels: ground-truth labels for texts batch of each identity in dataset_identity['orig_text']
        """

        # ----------group metric on orig_text----------
        # datas = {'text': dataset_identity['orig_text'], 'label': dataset_identity['label']}
        # dataset = DataFrame(datas)
        texts = dataset_identity['orig_text'].tolist()
        labels = dataset_identity['label'].tolist()
        active = nn.Sigmoid()

        # # ----------group metric on all_texts----------
        # texts = []
        # labels = []
        # for i in range(len(dataset_identity['orig_text'])):
        #     orig_text = dataset_identity['orig_text'][i]
        #     idis = eval(dataset_identity['idis'][i])
        #     label = dataset_identity['label'][i]
        #     texts.append(orig_text)
        #     labels.append(label)
        #     for idi in idis:
        #         texts.append(idi)
        #         labels.append(label)

        inputs = tokenizer(texts, padding=padding, max_length=max_seq_length, truncation=True,
                           return_tensors="pt")
        inputs.to(device)
        base = self.CustomModel.get_base_bertmodel_output(inputs)
        if model != None:
            output = model.forward(base)
        else:
            output = base64
        if classifier != None:
            logits = classifier(output)
        else:
            logits = self.CustomModel.custom_model['classifier'](output)
        overall_logits = active(logits)
        overall_labels = [pro.index(max(pro)) for pro in overall_logits.tolist()]

        all_texts = []
        all_base_labels = []
        for j in range(0, len(ID.identities)):
            identity = ID.identities[j]
            texts_identity = []
            label_identity = []
            for i in range(0, len(texts)):
                text = texts[i]
                if ID.identity_detect(text, identity) == True:
                    texts_identity.append(text)
                    label_identity.append(labels[i])
            all_texts.append(texts_identity)
            all_base_labels.append(label_identity)

        # prediction output for each subgroup withing one identity
        every_logits = []
        every_labels = []
        for i in range(0, len(all_texts)):
            # identity = ID.identities[i]
            texts = all_texts[i]
            if len(texts) == 0:
                every_logits.append(None)
                every_labels.append(None)
                continue
            inputs = tokenizer(texts, padding=padding, max_length=max_seq_length, truncation=True,
                               return_tensors="pt")
            inputs.to(device)
            base = self.CustomModel.get_base_bertmodel_output(inputs)
            if model != None:
                output = model.forward(base)
            else:
                output = base64
            if classifier != None:
                logits = classifier(output)
            else:
                logits = self.CustomModel.custom_model['classifier'](output)
            logits = active(logits)
            labels = [pro.index(max(pro)) for pro in logits.tolist()]
            every_logits.append(logits)
            every_labels.append(labels)
        # return every_logits, every_labels, all_base_labels
        return every_logits, every_labels, overall_logits, overall_labels, all_base_labels

    def get_predictions_all_basic(self, orig_classifier, dataset, tokenizer, padding, max_seq_length):
        active = nn.Sigmoid()
        texts = dataset.iloc[:, 0].values.tolist()
        inputs = tokenizer(texts, padding=padding, max_length=max_seq_length, truncation=True,
                           return_tensors="pt")
        inputs.to(device)
        base = self.CustomModel.get_base_bertmodel_output(inputs)
        logits = orig_classifier(base)
        logits = active(logits)
        # labels = [pro.index(max(pro)) for pro in logits.tolist()]
        return logits


    def get_predictions_all_idi(self, model, classifier, dataset_identity, tokenizer, padding, max_seq_length):
        all_logits = []
        all_labels = []
        num_columns = dataset_identity.shape[1]
        for i in range(0, num_columns - 1):
            texts = dataset_identity.iloc[:, i].values.tolist()
            inputs = tokenizer(texts, padding=padding, max_length=max_seq_length, truncation=True,
                               return_tensors="pt")
            inputs.to(device)
            # print("-->inputs[0]", np.array(inputs[0]))
            base = self.CustomModel.get_base_bertmodel_output(inputs)
            output = model.forward(base)
            # logits = self.CustomModel.custom_model['classifier'](output)
            logits = classifier(output)
            active = nn.Sigmoid()
            logits = active(logits)
            labels = [pro.index(max(pro)) for pro in logits.tolist()]
            all_logits.append(logits)
            all_labels.append(labels)
        return all_logits, all_labels

    def get_predictions_single(self, model, classifier, dataset, tokenizer, padding, max_seq_length, if_identity=False):
        """
        predict all texts in dataset once as a whole brunch
        if model == None: no added linear function, use base_model to predict sampels
        if_identity = True: further output identity information
        """
        active_softmax = nn.Softmax()
        texts = dataset['text'].values.tolist()
        all_logits = []
        all_labels = []
        identities_result = []

        for text in texts: #tqdm

            input = tokenizer(text, padding=padding, max_length=max_seq_length, truncation=True,
                               return_tensors="pt")
            input.to(device)
            base = self.CustomModel.get_base_bertmodel_output(input)
            try:
                output = model.forward(base)
            except:
                output = base
            if classifier != None:
                logits = classifier(output)
            else:
                logits = self.CustomModel.custom_model['classifier'](output)
            labels = [pro.index(max(pro)) for pro in logits.tolist()]
            # print("-->probability_M", probability_M)
            prob_logits = active_softmax(logits)
            all_logits.append(prob_logits.tolist()[0])
            all_labels.append(labels[0])

        logits = all_logits
        labels = all_labels
        if if_identity == True:
            return logits, labels, identities_result
        else:
            return logits, labels

    # def get_predictions_all(self, model, debias_dataset, tokenizer, padding, max_seq_length):
    #     text_all_M = debias_dataset['text_all_M'].values.tolist()
    #     text_all_F = debias_dataset['text_all_F'].values.tolist()
    #     print("-->text_all_M", debias_dataset['text_all_M'])
    #     inputs_M = tokenizer(text_all_M, padding=padding, max_length=max_seq_length, truncation=True,
    #                          return_tensors="pt")
    #     inputs_F = tokenizer(text_all_F, padding=padding, max_length=max_seq_length, truncation=True,
    #                          return_tensors="pt")
    #     inputs_M.to(device)
    #     inputs_F.to(device)
    #     # print("-->model(**inputs)", self.CustomModel.base_model(**inputs_M))
    #     base_M = self.CustomModel.get_base_bertmodel_output(inputs_M)
    #     print("-->base_M", base_M)
    #     output_M = model.forward(base_M)
    #     print("-->output_M", output_M)
    #     # logits_M = self.CustomModel.custom_model['classifier'](output_M)
    #     # print("-->logits_M", logits_M)
    #     logits_M = self.CustomModel.custom_model['classifier'](output_M)
    #     probability_M = logits_M[:, 1]
    #     labels_M = [pro.index(max(pro)) for pro in logits_M.tolist()]
    #     # print("-->probability_M", probability_M)
    #
    #     base_F = self.CustomModel.get_base_bertmodel_output(inputs_F)
    #     output_F = model.forward(base_F)
    #     logits_F = self.CustomModel.custom_model['classifier'](output_F)
    #     probability_F = logits_F[:, 1]
    #     labels_F = [pro.index(max(pro)) for pro in logits_F.tolist()]
    #     # print("-->probability_F", probability_F)
    #
    #     return probability_M, probability_F, logits_M, logits_F

    def get_predictions_all(self, model, classifier, dataset, tokenizer, padding, max_seq_length):
        try:
            texts = dataset['orig_text'].tolist()
        except:
            texts = dataset['text'].tolist()
        # labels = dataset['label'].tolist()
        active = nn.Sigmoid()

        inputs = tokenizer(texts, padding=padding, max_length=max_seq_length, truncation=True,
                           return_tensors="pt")
        inputs.to(device)
        base = self.CustomModel.get_base_bertmodel_output(inputs)
        if model != None:
            output = model.forward(base)
        else:
            output = base
        if classifier != None:
            logits = classifier(output)
        else:
            logits = self.CustomModel.custom_model['classifier'](output)

        logits = active(logits)
        predict_labels = [pro.index(max(pro)) for pro in logits.tolist()]

        return logits, predict_labels


    def get_predictions_with_identity(self, model, dataset, tokenizer, padding, max_seq_length):
        text = dataset['text'].values.tolist()
        # print("-->text", text)
        # print(type(text))
        inputs = tokenizer(text, padding=padding, max_length=max_seq_length, truncation=True,
                             return_tensors="pt")
        inputs.to(device)
        base = self.CustomModel.get_base_bertmodel_output(inputs)
        output = model.forward(base)
        logits = self.CustomModel.custom_model['classifier'](output)
        probability = logits[:, 1]
        labels = [pro.index(max(pro)) for pro in logits.tolist()]

        all_results = [[]]*len(identities)
        for i in range(len(identities)):
            identity = identities[i]
            # print("-->i:{}, identity:{}".format(i, identity))
            for j in range(len(text)):
                t = text[j]
                if identity_detect(t, identity) == True:
                    # print("text:{}, identity:{}".format(t, identity))
                    all_results[i] = all_results[i] + [j]
        return probability, labels, all_results

    def AUC_calculation(self, logits, true_labels):
        auc_score = roc_auc_score(true_labels, logits)
        return auc_score


    # def get_predictions(self, model, dataset, tokenizer, padding, max_seq_length):

    def loss_function(self, debias_dataset, tokenizer, padding, max_seq_length):
        score = self.metric_calculation(debias_dataset, tokenizer, padding, max_seq_length)
        return torch.tensor(score)

    def loss_funtion_diff(self, x1, x2):
        score = torch.mean(torch.abs(x1 - x2))
        print("-->score", score)
        return torch.mean(score)


    # weighted multi-class log loss
    def loss_function_log_weight(y_true, y_pred):
        epsilon = 1e-7  # Define epsilon so that the backpropagation will not result in NaN for 0 divisor case
        n_classes = 2  # As there are two classes in the dataset
        weights = np.ones(n_classes,
                          dtype='float32')  # For this competition, you can use weights from probing the leaderboard
        print("-->weights", weights)
        class_counts = df.groupby('label')['image_id'].count().values
        class_proportions = class_counts / np.max(class_counts)
        K.set_floatx('float32')  # You can also set backend to any float

        # Clipping the prediction value
        y_pred_clipped = K.clip(y_pred, epsilon, 1 - epsilon)
        # true labels weighted by weights and percent elements per class
        y_true_weighted = (y_true * weights) / class_proportions
        # multiply tensors element-wise and then sum
        loss_num = (y_true_weighted * K.log(y_pred_clipped))
        loss = -1 * K.sum(loss_num) / K.sum(weights)
        return loss

    def loss_function_weighted_logloss(self, y_true, y_pred,  w_0=5.0, w_1=1.0, eps=1e-15):
        # how to generate tensor labels
        # labels = torch.tensor(dataset_one_batch['label'].values.tolist())
        # labels = torch.tensor([[(l + 1) % 2, (l + 2) % 2] for l in labels], device=device)
        # labels.to(device)

        y_true = y_true.float()
        # print(type(y_true), type(y_pred))

        # Clip y_pred between eps and 1-eps
        # p = np.clip(y_pred, eps, 1 - eps)
        # loss = np.sum(- w_0 * y_true * np.log(p) - w_1 * (1 - y_true) * np.log(1 - p))

        weights = torch.tensor([w_0, w_1], device=device)
        criterian = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
        # criterian = nn.BCEWithLogitsLoss(reduction='mean')
        loss = criterian(y_pred, y_true)

        return loss

    def loss_function_sprt(self, y_true, y_pred):
        y_true = y_true.float()
        # original SPRT_ratio calculation
        accept_bound = torch.tensor((1 - beta) / alpha)
        deny_bound = torch.tensor(beta / (1 - alpha))
        p0 = self.threshold - self.sigma
        p1 = self.threshold + self.sigma
        sprt_ratio = torch.tensor((pow(p1, s) * pow(1 - p1, n - s)) / (pow(p0, s) * pow(1 - p0, n - s)))


    def log_loss_value(Z, weights, total_weights, rho):
        """
        Parameters
        ----------
        Z               numpy.array containing training data with shape = (n_rows, n_cols)
        rho             numpy.array of coefficients with shape = (n_cols,)
        total_weights   numpy.sum(total_weights) (only included to reduce computation)
        weights         numpy.array of sample weights with shape (n_rows,)

        Returns
        -------
        loss_value  scalar = 1/n_rows * sum(log( 1 .+ exp(-Z*rho))

        """
        scores = Z.dot(rho)
        pos_idx = scores > 0
        loss_value = np.empty_like(scores)
        loss_value[pos_idx] = np.log1p(np.exp(-scores[pos_idx]))
        loss_value[~pos_idx] = -scores[~pos_idx] + np.log1p(np.exp(scores[~pos_idx]))
        loss_value = loss_value.dot(weights) / total_weights
        return loss_value

        # todo: loss_function = accuracy+diff
    def loss_function_with_acc(self, x1, x2, logits1, logits2, base_l):
        criterion = nn.MSELoss()
        bias_loss = torch.mean(torch.abs(x1 - x2))
        base_l = base_l.float()
        # acc_loss1 = F.mse_loss(logits1.squeeze(), base_l.squeeze())
        acc_loss1 = criterion(logits1.squeeze(), base_l.squeeze())
        acc_loss2 = criterion(logits2.squeeze(), base_l.squeeze())
        # acc_loss = torch.mean([acc_loss1, acc_loss2])
        # print("-->acc_loss", acc_loss)
        acc_loss = torch.div(torch.sum(acc_loss1 + acc_loss2), 2.0)
        acc_loss = acc_loss.to(torch.float32)
        loss = torch.sum(bias_loss + acc_loss)
        return loss, bias_loss, acc_loss

    def loss_function_fpr_subgroup(self, predict_labels, base_l, metric):
        criterion = nn.MSELoss()
        base_l = base_l.float()
        acc_loss = criterion(predict_labels, base_l)
        bias_loss = torch.tensor(metric)
        # loss = acc_loss + bias_loss
        loss = metric
        return loss, bias_loss, acc_loss

    def loss_function_sprt_subgroup(self, predict_labels, base_l, s, n):
        criterion = nn.MSELoss()
        base_l = base_l.float()
        # acc_loss = criterion(predict_labels.squeeze(), base_l.squeeze())
        # print("-->predict_labels", predict_labels, len(predict_labels))
        # print("-->base_l", base_l, len(base_l))
        acc_loss = criterion(predict_labels, base_l)

        accept_bound = torch.tensor((1 - beta) / alpha)
        deny_bound = torch.tensor(beta / (1 - alpha))
        p0 = self.threshold - self.sigma
        p1 = self.threshold + self.sigma

        # if bias_metric <= threshold:
        #     s += 1
        sprt_ratio = torch.tensor((pow(p1, s)*pow(1-p1, n-s))/(pow(p0, s)*pow(1-p0, n-s)))
        if sprt_ratio >= accept_bound:
            print("-->sprt_ratio:{}>=accept_bound:{}".formate(sprt_ratio, accept_bound))
            # return True
            bias_loss = torch.tensor(0)
        else:
            bias_loss = accept_bound - sprt_ratio
        loss = acc_loss + bias_loss
        # loss = bias_loss
        return loss, bias_loss, acc_loss


    def loss_function_FNR(self, y_true, y_pred, orig_labels, label_0, label_1, actual_P):
        """
        FNR = FN/actual P
        """
        # print("-->y_true", y_true)
        # print(y_true.grad_fn)
        # print("-->y_pred", y_pred)
        # print(y_pred.grad_fn)
        # print("-->orig_labels", orig_labels)

        predict_labels = [pro.index(max(pro)) for pro in y_pred.tolist()]
        # print("-->pred_labels", predict_labels)

        criterian = nn.MSELoss()
        acc_loss = criterian(y_pred, y_true)

        criterian_each = nn.MSELoss(reduction='none')
        active = nn.Sigmoid()

        # actual_P = torch.sum(orig_labels)
        # print("-->actual_P", actual_P)

        pred_0_loss = torch.mean(criterian_each(y_pred, label_0), 1)
        pred_1_loss = torch.mean(criterian_each(y_pred, label_1), 1)
        # print("-->pred_0_loss", pred_0_loss)
        # print(pred_0_loss.grad_fn)
        # print("-->pred_1_loss", pred_0_loss)
        # print(pred_1_loss.grad_fn)
        # print("-->criterian(y_pred, label_0)", criterian(y_pred, label_0))
        # print("-->criterian(y_pred, label_1)", criterian(y_pred, label_1))
        minus = torch.sub(pred_0_loss, pred_1_loss)
        # print(minus.grad_fn)
        pred_label = active(torch.mul(minus, 10000))
        # print("-->pred_label", pred_label)
        # print(predict_N.grad_fn)

        FN = torch.sum(active(torch.mul(torch.sub(orig_labels, pred_label), 10000)))
        print("-->FN:{}, actual_P:{}".format(FN, actual_P))
        # print(FN.grad_fn)

        FNR = torch.div(FN, actual_P).float()
        # print("-->FNR", FNR)
        # print(FNR.grad_fn)

        # return FNR, FNR, FNR

        # loss = torch.add(acc_loss, torch.mul(FNR, 0.5))
        loss = torch.add(acc_loss, FNR)
        return loss, acc_loss, FNR

    def loss_function_MSE(self, y_true, y_pred):
        criterian = nn.MSELoss()
        acc_loss = criterian(y_pred, y_true)

        return acc_loss

    def normalization(self, input, min_value, max_value):
        """
        normalized_score = (input - min_value) / (max_value - min_value)
        """
        active_relu = nn.ReLU()

        min_value = torch.tensor(min_value, device=device).float()
        max_value = torch.tensor(max_value, device=device).float()
        normalized_score = torch.div(torch.sub(input, min_value), torch.sub(max_value, min_value))
        normalized_score = - active_relu(- normalized_score + 1) + 1
        return normalized_score

    def loss_function_sprt_individual(self, truth_labels, all_logits, all_labels, base_logits, MinValue, MaxValue, gamma):
        """
                orig_labels = ground_truth label
                pred_labels = predicted labels for orig_texts
                H0: individual fair
                H1: individualb unfair
                s: number of individual's whose label not change after identity changes
                n: all samples
                """
        # MaxValue = (math.sqrt(self.batch_size) * (max_p - self.threshold)) / \
        #            (math.sqrt(self.threshold * (1 - self.threshold)))
        # MinVaue = (math.sqrt(self.batch_size) * (min_p - self.threshold)) / \
        #            (math.sqrt(self.threshold * (1 - self.threshold)))

        pred_labels = all_labels[0]
        logits_orig = all_logits[0]
        logits_idi1 = all_logits[1]
        logits_idi2 = all_logits[2]
        logits_idi3 = all_logits[3]
        logits_idi4 = all_logits[4]
        logits_idi5 = all_logits[5]

        criterian_each = nn.MSELoss(reduction='none')
        active = nn.Sigmoid()
        active_relu = nn.ReLU()
        criterian = nn.BCELoss(reduction='mean')
        # criterian = nn.BCEWithLogitsLoss(reduction='mean')

        # acc_loss
        labels = torch.tensor([[(l + 1) % 2, (l + 2) % 2] for l in truth_labels], device=device)
        labels = labels.float()
        acc_loss = criterian(logits_orig, labels)  # acc_loss on predictions with editor and updated classifier
        # normalization
        acc_loss = self.normalization(acc_loss, 0.3, 0.8)
        # base_acc_loss = criterian(base_logits, labels)  # acc_loss on predictions with original BERT
        # print("-->base_acc_loss:{}, acc_loss:{}".format(base_acc_loss, acc_loss))
        # # normalization
        # new_acc_loss = self.normalization(acc_loss, base_acc_loss, 1)
        # print("-->new_acc_loss", new_acc_loss)
        # # acc_loss = self.normalization(active_relu(acc_loss - base_acc_loss), 0, 1 - base_acc_loss)

        # _, predicted_labels = torch.max(torch.nn.functional.softmax(logits_orig, dim=1), dim=1)

        acc = accuracy_score(truth_labels, pred_labels)
        print("-->ACC", acc)
        f1 = f1_score(truth_labels, pred_labels)
        print("-->F1", f1)

        label_true = torch.tensor([[(l + 1) % 2, (l + 2) % 2] for l in pred_labels], device=device).float()
        label_false = torch.tensor([[(l + 2) % 2, (l + 1) % 2] for l in pred_labels], device=device).float()

        # idi1
        pred_true_loss = torch.mean(criterian_each(logits_idi1, label_true), 1)
        pred_false_loss = torch.mean(criterian_each(logits_idi1, label_false), 1)
        minus = torch.sub(pred_true_loss, pred_false_loss)
        minus_relu = active_relu(minus)

        # idi2
        pred_true_loss = torch.mean(criterian_each(logits_idi2, label_true), 1)
        pred_false_loss = torch.mean(criterian_each(logits_idi2, label_false), 1)
        minus = torch.sub(pred_true_loss, pred_false_loss)
        minus_relu = minus_relu + active_relu(minus)

        # idi3
        pred_true_loss = torch.mean(criterian_each(logits_idi3, label_true), 1)
        pred_false_loss = torch.mean(criterian_each(logits_idi3, label_false), 1)
        minus = torch.sub(pred_true_loss, pred_false_loss)
        minus_relu = minus_relu + active_relu(minus)

        # idi4
        pred_true_loss = torch.mean(criterian_each(logits_idi4, label_true), 1)
        pred_false_loss = torch.mean(criterian_each(logits_idi4, label_false), 1)
        minus = torch.sub(pred_true_loss, pred_false_loss)
        minus_relu = minus_relu + active_relu(minus)

        # idi5
        pred_true_loss = torch.mean(criterian_each(logits_idi5, label_true), 1)
        pred_false_loss = torch.mean(criterian_each(logits_idi5, label_false), 1)
        minus = torch.sub(pred_true_loss, pred_false_loss)
        minus_relu = minus_relu + active_relu(minus)

        pred_false = active(torch.mul(minus_relu, 100)).float()  # 500

        label_05 = torch.tensor([0.5 for l in truth_labels], device=device).float()
        pred_if_false = torch.mul(torch.sub(pred_false, label_05), 2).float()

        # label_1 = torch.tensor([1 for l in truth_labels], device=device).float()
        # pred_if_true = torch.sub(label_1, pred_if_false)

        sum_if_false = torch.sum(pred_if_false)
        n = torch.tensor(logits_orig.size(0), device=device).float()
        p0 = torch.tensor(self.threshold, device=device).float()
        p_hat = torch.div(sum_if_false, n)
        # p_hat = torch.div(sum_if_false, logits_orig.size(0))

        # z_score = active_relu(torch.div(torch.mul(torch.sqrt(n), torch.sub(p_hat, self.threshold)),
        #                     torch.sqrt(torch.mul(p0, torch.sub(1, p0)))))
        z_score = torch.div(torch.mul(torch.sqrt(n), torch.sub(p_hat, self.threshold)),
                                        torch.sqrt(torch.mul(p0, torch.sub(1, p0))))

        print("-->z_score:{}, idi:{}, p_hat:{}, n:{}".format(z_score, sum_if_false, p_hat, n))

        bias_loss = self.normalization(z_score, MinValue, MaxValue)
        # gamma = torch.tensor(self.gamma, device=device).float()
        gamma = torch.tensor(gamma, device=device).float()
        bias_loss = torch.mul(bias_loss, gamma).float()
        loss = torch.add(acc_loss, bias_loss)

        return loss, acc_loss, bias_loss

    def loss_function_sprt_fpr(self, orig_labels, logits_overall, identity_num, show_identity_num, MinValue, MaxValue, logits_identity,
                               baseline_fur_train, gamma=None):
        """
        INPUTS:
                    orig_labels: ground-truth labels of dataset_identity_one_batch
                    logits_overall: prediction logits for acc_loss calculation
                    identity_num: the number of samples of dataset_one_batch['text'] for each identity
                    show_identity_num: the number of identities shown in this batch
                    MinValue: minimum value of z_score
                    MaxValue: maximum value of z_score
                    logits_identity: a list, prediction logits for each identity in dataset_identity_one_batch['text']
                    baseline_fur_train: the baseline FUR
        FNR = FN/actual P
        CALCULATION: MEAN(every|FNR_s - baseline_FUR|)
        """

        criterian_each = nn.MSELoss(reduction='none')
        active = nn.Sigmoid()
        active_relu = nn.ReLU()
        criterian = nn.BCELoss(reduction='mean')

        # acc_loss
        labels = torch.tensor([[(l + 1) % 2, (l + 2) % 2] for l in orig_labels], device=device)
        labels = labels.float()
        acc_loss = criterian(logits_overall, labels)  # acc_loss on predictions with editor and updated classifier
        # normalization
        # acc_loss = self.normalization(acc_loss, 0.3, 0.8)

        # bias_loss
        label_1 = torch.tensor([[0, 1] for i in range(0, identity_num[0])], device=device).float()
        label_0 = torch.tensor([[1, 0] for i in range(0, identity_num[0])], device=device).float()
        label_1.to(device)
        label_0.to(device)
        # male
        pred_0_loss = torch.mean(criterian_each(logits_identity[0], label_0), 1)
        pred_1_loss = torch.mean(criterian_each(logits_identity[0], label_1), 1)
        minus = torch.sub(pred_0_loss, pred_1_loss)
        pred_label = active(torch.mul(minus, 800)).float()  # 1 if predict label==1
        fn = torch.sum(pred_label)
        # FN_overall = fn
        FNR = torch.div(fn, identity_num[0]).float()
        print("-->FNR:{}, FN:{}, P:{}".format(FNR, fn, identity_num[0]))

        baseline_fur_train = torch.tensor(baseline_fur_train).float()
        baseline_fur_train.to(device)
        DIFF_overall = torch.abs(torch.sub(FNR, baseline_fur_train))

        # truth_labels = torch.tensor(identity_labels[0], device=device).float()
        # FN_list = active_relu(torch.sub(truth_labels, pred_label))  # 1 if FN
        # fn = torch.sum(FN_list)
        # weight = P_overall / float(P_identity[0])
        # FN_regulated = torch.mul(fn, torch.tensor(weight, device=device).float())
        # FNR = torch.div(fn, P_identity[0]).float()
        # print("-->FNR:{}, FN:{}, P:{}".format(FNR, fn, P_identity[0]))


        # female homosexual christian muslim jewish black white
        for i in range(1, len(logits_identity)):
            logits = logits_identity[i]
            if logits == None:
                print("-->None")
                continue
            label_1 = torch.tensor([[0, 1] for i in range(0, identity_num[i])], device=device).float()
            label_0 = torch.tensor([[1, 0] for i in range(0, identity_num[i])], device=device).float()
            label_1.to(device)
            label_0.to(device)
            pred_0_loss = torch.mean(criterian_each(logits, label_0), 1)
            pred_1_loss = torch.mean(criterian_each(logits, label_1), 1)
            minus = torch.sub(pred_0_loss, pred_1_loss)
            pred_label = active(torch.mul(minus, 800)).float()  # 1 if predict label==1  # 500
            fn = torch.sum(pred_label)
            FNR = torch.div(fn, identity_num[i]).float()
            print("-->FNR:{}, FN:{}, P:{}".format(FNR, fn, identity_num[i]))
            DIFF_overall = DIFF_overall + torch.abs(torch.sub(FNR, baseline_fur_train))

        # div not all identity_num but the identity_num shown
        show_identity_num = torch.tensor(show_identity_num).float()
        show_identity_num.to(device)
        p_hat = torch.div(DIFF_overall, show_identity_num).float()
        # n = torch.tensor(logits_overall.size(0), device=device).float()
        n = torch.tensor(self.batch_size, device=device).float()
        p0 = torch.tensor(self.threshold, device=device).float()

        z_score = torch.div(torch.mul(torch.sqrt(n), torch.sub(p_hat, self.threshold)),
                            torch.sqrt(torch.mul(p0, torch.sub(1, p0))))

        print("-->z_score:{}, p_hat:{}, n:{}".format(z_score, p_hat, n))

        bias_loss = self.normalization(z_score, MinValue, MaxValue)
        if gamma == None:
            gamma = torch.tensor(self.gamma, device=device).float()
        else:
            gamma = torch.tensor(gamma, device=device).float()
        bias_loss = torch.mul(bias_loss, gamma).float()
        loss = torch.add(acc_loss, bias_loss)

        return loss, acc_loss, bias_loss

    def loss_function_sprt_fpr1(self, orig_labels, logits_overall, identity_num, show_identity_num, MinValue, MaxValue, logits_identity,
                               baseline_fur_train, gamma=None):
        """
        CALCULATION: MEAN(every(FNR_s))
        """
        criterian_each = nn.MSELoss(reduction='none')
        active = nn.Sigmoid()
        active_relu = nn.ReLU()
        criterian = nn.BCELoss(reduction='mean')

        # acc_loss
        labels = torch.tensor([[(l + 1) % 2, (l + 2) % 2] for l in orig_labels], device=device)
        labels = labels.float()
        acc_loss = criterian(logits_overall, labels)  # acc_loss on predictions with editor and updated classifier
        # normalization
        # acc_loss = self.normalization(acc_loss, 0.3, 0.8)

        # bias_loss
        label_1 = torch.tensor([[0, 1] for i in range(0, identity_num[0])], device=device).float()
        label_0 = torch.tensor([[1, 0] for i in range(0, identity_num[0])], device=device).float()
        label_1.to(device)
        label_0.to(device)
        # male
        pred_0_loss = torch.mean(criterian_each(logits_identity[0], label_0), 1)
        pred_1_loss = torch.mean(criterian_each(logits_identity[0], label_1), 1)
        minus = torch.sub(pred_0_loss, pred_1_loss)
        pred_label = active(torch.mul(minus, 800)).float()  # 1 if predict label==1
        fn = torch.sum(pred_label)
        # FN_overall = fn
        FNR = torch.div(fn, identity_num[0]).float()
        print("-->FNR:{}, FN:{}, P:{}".format(FNR, fn, identity_num[0]))

        FUR_overall = FNR

        # baseline_fur_train = torch.tensor(baseline_fur_train).float()
        # baseline_fur_train.to(device)
        # DIFF_overall = torch.abs(torch.sub(FNR, baseline_fur_train))

        # female homosexual christian muslim jewish black white
        for i in range(1, len(logits_identity)):
            logits = logits_identity[i]
            if logits == None:
                print("-->None")
                continue
            label_1 = torch.tensor([[0, 1] for i in range(0, identity_num[i])], device=device).float()
            label_0 = torch.tensor([[1, 0] for i in range(0, identity_num[i])], device=device).float()
            label_1.to(device)
            label_0.to(device)
            pred_0_loss = torch.mean(criterian_each(logits, label_0), 1)
            pred_1_loss = torch.mean(criterian_each(logits, label_1), 1)
            minus = torch.sub(pred_0_loss, pred_1_loss)
            pred_label = active(torch.mul(minus, 800)).float()  # 1 if predict label==1  # 500
            fn = torch.sum(pred_label)
            FNR = torch.div(fn, identity_num[i]).float()
            print("-->FNR:{}, FN:{}, P:{}".format(FNR, fn, identity_num[i]))
            FUR_overall = FUR_overall + FNR
            # DIFF_overall = DIFF_overall + torch.abs(torch.sub(FNR, baseline_fur_train))

        # div not all identity_num but the identity_num shown
        show_identity_num = torch.tensor(show_identity_num).float()
        show_identity_num.to(device)
        # p_hat = torch.div(DIFF_overall, show_identity_num).float()
        p_hat = torch.div(FUR_overall, show_identity_num).float()
        # n = torch.tensor(logits_overall.size(0), device=device).float()
        n = torch.tensor(self.batch_size, device=device).float()
        p0 = torch.tensor(self.threshold, device=device).float()

        z_score = torch.div(torch.mul(torch.sqrt(n), torch.sub(p_hat, self.threshold)),
                            torch.sqrt(torch.mul(p0, torch.sub(1, p0))))

        print("-->z_score:{}, p_hat:{}, n:{}".format(z_score, p_hat, n))

        bias_loss = self.normalization(z_score, MinValue, MaxValue)
        if gamma == None:
            gamma = torch.tensor(self.gamma, device=device).float()
        else:
            gamma = torch.tensor(gamma, device=device).float()
        bias_loss = torch.mul(bias_loss, gamma).float()
        loss = torch.add(acc_loss, bias_loss)

        return loss, acc_loss, bias_loss

    def loss_function_sprt_fur(self, orig_labels, orig_identity_labels, logits_overall, identity_num, show_identity_num, MinValue, MaxValue, logits_identity,
                               baseline_fur_train, all_labels, gamma=None):
        """
        INPUTS:
                    orig_labels: ground-truth labels of dataset_identity_one_batch
                    orig_identity_labels: ground_truth labels of each subgroup
                    logits_overall: prediction logits for acc_loss calculation
                    identity_num: the number of samples of dataset_one_batch['text'] for each identity
                    show_identity_num: the number of identities shown in this batch
                    MinValue: minimum value of z_score
                    MaxValue: maximum value of z_score
                    logits_identity: a list, prediction logits for each identity in dataset_identity_one_batch['text']
                    baseline_fur_train: the baseline FUR
        DIFFERENCE: no pre-processing (extract G_Positive withing loss function)
        CALCULATION: MEAN(every|FNR_s - baseline_FUR|)
        """

        criterian_each = nn.MSELoss(reduction='none')
        active = nn.Sigmoid()
        active_relu = nn.ReLU()
        criterian = nn.BCELoss(reduction='mean')

        # acc_loss
        labels = torch.tensor([[(l + 1) % 2, (l + 2) % 2] for l in orig_labels], device=device)
        labels = labels.float()
        acc_loss = criterian(logits_overall, labels)  # acc_loss on predictions with editor and updated classifier
        # normalization
        # acc_loss = self.normalization(acc_loss, 0.3, 0.8)

        # bias_loss
        label_1 = torch.tensor([[0, 1] for i in range(0, identity_num[0])], device=device).float()
        label_0 = torch.tensor([[1, 0] for i in range(0, identity_num[0])], device=device).float()
        label_1.to(device)
        label_0.to(device)
        # male
        pred_0_loss = torch.mean(criterian_each(logits_identity[0], label_0), 1)
        pred_1_loss = torch.mean(criterian_each(logits_identity[0], label_1), 1)
        minus = torch.sub(pred_0_loss, pred_1_loss)
        pred_labels = active(torch.mul(minus, 800)).float()  # 1 if predict label == 1

        # extract false positive predicted labels
        truth_labels = torch.tensor(orig_identity_labels[0], device=device).float()
        if self.privileged_label == 1:  # FUR = FNR
            fu_list = active_relu(torch.sub(truth_labels, pred_labels))  # 1 if false negative
        else:   # FUR = FPR (self.privileged_label = 0)
            fu_list = active_relu(torch.sub(pred_labels, truth_labels))  # 1 if false positive
        fu = torch.sum(fu_list)

        P = torch.sum(truth_labels == self.privileged_label)
        FUR = torch.div(fu, P).float()
        print("-->FUR:{}, FU:{}, P:{}".format(FUR, fu,P))

        # CHECK
        # print("CHECK")
        # if self.privileged_label == 1:  # FUR = FNR
        #     tp, fp, tn, fn = self.perf_measure(orig_identity_labels[0], all_labels[0])
        #     print(orig_identity_labels[0])
        #     fpr = fp / float(fp + tn)
        #     fnr = fn / float(fn + tp)
        #     print("-->FUR:{}, FU:{}, P:{}".format(fnr, fn, fn + tp))
        # else:   # FUR = FPR (self.privileged_label = 0)
        #     tp, fp, tn, fn = self.perf_measure(orig_identity_labels[0], all_labels[0])
        #     fpr = fp / float(fp + tn)
        #     fnr = fn / float(fn + tp)
        #     print("-->FUR:{}, FU:{}, P:{}".format(fpr, fp, fp + tn))

        baseline_fur_train = torch.tensor(baseline_fur_train).float()
        baseline_fur_train.to(device)
        DIFF_overall = torch.abs(torch.sub(FUR, baseline_fur_train))

        # female homosexual christian muslim jewish black white
        for i in range(1, len(logits_identity)):
            logits = logits_identity[i]
            if logits == None:
                print("-->None")
                continue
            label_1 = torch.tensor([[0, 1] for i in range(0, identity_num[i])], device=device).float()
            label_0 = torch.tensor([[1, 0] for i in range(0, identity_num[i])], device=device).float()
            label_1.to(device)
            label_0.to(device)
            pred_0_loss = torch.mean(criterian_each(logits, label_0), 1)
            pred_1_loss = torch.mean(criterian_each(logits, label_1), 1)
            minus = torch.sub(pred_0_loss, pred_1_loss)
            pred_labels = active(torch.mul(minus, 800)).float()   # 1 if predict label==1  # 500

            # extract false positive predicted labels
            truth_labels = torch.tensor(orig_identity_labels[i], device=device).float()
            if self.privileged_label == 1:  # FUR = FNR
                fu_list = active_relu(torch.sub(truth_labels, pred_labels))  # 1 if false negative
            else:  # FUR = FPR (self.privileged_label = 0)
                fu_list = active_relu(torch.sub(pred_labels, truth_labels))  # 1 if false positive
            fu = torch.sum(fu_list)

            P = torch.sum(truth_labels == self.privileged_label)
            FUR = torch.div(fu, P).float()
            print("-->FUR:{}, FU:{}, P:{}".format(FUR, fu, P))
            # CHECK
            # print("CHECK")
            # if self.privileged_label == 1:  # FUR = FNR
            #     tp, fp, tn, fn = self.perf_measure(orig_identity_labels[i], all_labels[i])
            #     print(orig_identity_labels[i])
            #     fpr = fp / float(fp + tn)
            #     fnr = fn / float(fn + tp)
            #     print("-->FUR:{}, FU:{}, P:{}".format(fnr, fn, fn + tp))
            # else:  # FUR = FPR (self.privileged_label = 0)
            #     tp, fp, tn, fn = self.perf_measure(orig_identity_labels[i], all_labels[i])
            #     fpr = fp / float(fp + tn)
            #     fnr = fn / float(fn + tp)
            #     print("-->FUR:{}, FU:{}, P:{}".format(fpr, fp, fp + tn))

            DIFF_overall = DIFF_overall + torch.abs(torch.sub(FUR, baseline_fur_train))

        # div not all identity_num but the identity_num shown
        show_identity_num = torch.tensor(show_identity_num).float()
        show_identity_num.to(device)
        p_hat = torch.div(DIFF_overall, show_identity_num).float()
        # n = torch.tensor(logits_overall.size(0), device=device).float()
        n = torch.tensor(self.batch_size, device=device).float()
        p0 = torch.tensor(self.threshold, device=device).float()

        z_score = torch.div(torch.mul(torch.sqrt(n), torch.sub(p_hat, self.threshold)),
                            torch.sqrt(torch.mul(p0, torch.sub(1, p0))))

        print("-->z_score:{}, p_hat:{}, n:{}".format(z_score, p_hat, n))

        bias_loss = self.normalization(z_score, MinValue, MaxValue)
        if gamma == None:
            gamma = torch.tensor(self.gamma, device=device).float()
        else:
            gamma = torch.tensor(gamma, device=device).float()
        bias_loss = torch.mul(bias_loss, gamma).float()
        loss = torch.add(acc_loss, bias_loss)

        return loss, acc_loss, bias_loss


    def loss_function_individual(self, y_true, orig_labels,
                                      logits_male, logits_female, logits_homosexual, logits_christian, logits_muslim,
                                      logits_jewish, logits_black, logits_white, logits_illness):
        criterian = nn.MSELoss(reduce=True, size_average=False)
        criterian_each = nn.MSELoss(reduction='none')
        active = nn.Sigmoid()
        active_relu = nn.ReLU()
        active_softmax = nn.Softmax()
        acc_loss = criterian(logits_male, y_true)

        label_0 = torch.tensor([[1, 0] for l in orig_labels], device=device).float()
        label_1 = torch.tensor([[0, 1] for l in orig_labels], device=device).float()
        label_0.to(device)
        label_1.to(device)

        labels = [pro.index(max(pro)) for pro in logits_male.tolist()]
        # print("-->orig_labels", orig_labels.tolist())
        label_true = torch.tensor([[(l + 1) % 2, (l + 2) % 2] for l in labels], device=device).float()
        label_false = torch.tensor([[(l + 2) % 2, (l + 1) % 2] for l in labels], device=device).float()

        # male
        pred_true_loss = torch.mean(criterian_each(logits_male, label_true), 1)
        pred_false_loss = torch.mean(criterian_each(logits_male, label_false), 1)
        minus = torch.sub(pred_true_loss, pred_false_loss)
        minus_relu = active_relu(minus)
        # pred_if_false = active(torch.mul(minus, 10000))  # if pred_if_false == 1, pred_label != ground truth label

        # female
        pred_true_loss = torch.mean(criterian_each(logits_female, label_true), 1)
        pred_false_loss = torch.mean(criterian_each(logits_female, label_false), 1)
        minus = torch.sub(pred_true_loss, pred_false_loss)
        minus_relu = minus_relu + active_relu(minus)
        # pred_if_false = torch.add(pred_if_false, active(torch.mul(minus, 10000)))
        # print("-->pred_if_false", pred_if_false)

        # homosexual
        pred_true_loss = torch.mean(criterian_each(logits_homosexual, label_true), 1)
        pred_false_loss = torch.mean(criterian_each(logits_homosexual, label_false), 1)
        minus = torch.sub(pred_true_loss, pred_false_loss)
        minus_relu = minus_relu + active_relu(minus)
        # pred_if_false = torch.add(pred_if_false, active(torch.mul(minus, 10000)))
        # print("-->pred_if_false", pred_if_false)

        # christian
        pred_true_loss = torch.mean(criterian_each(logits_christian, label_true), 1)
        pred_false_loss = torch.mean(criterian_each(logits_christian, label_false), 1)
        minus = torch.sub(pred_true_loss, pred_false_loss)
        minus_relu = minus_relu + active_relu(minus)
        # pred_if_false = torch.add(pred_if_false, active(torch.mul(minus, 10000)))
        # print("-->pred_if_false", pred_if_false)

        # muslim
        pred_true_loss = torch.mean(criterian_each(logits_muslim, label_true), 1)
        pred_false_loss = torch.mean(criterian_each(logits_muslim, label_false), 1)
        minus = torch.sub(pred_true_loss, pred_false_loss)
        minus_relu = minus_relu + active_relu(minus)
        # pred_if_false = torch.add(pred_if_false, active(torch.mul(minus, 10000)))
        # print("-->pred_if_false", pred_if_false)

        # jewish
        pred_true_loss = torch.mean(criterian_each(logits_jewish, label_true), 1)
        pred_false_loss = torch.mean(criterian_each(logits_jewish, label_false), 1)
        minus = torch.sub(pred_true_loss, pred_false_loss)
        minus_relu = minus_relu + active_relu(minus)
        # pred_if_false = torch.add(pred_if_false, active(torch.mul(minus, 10000)))
        # print("-->pred_if_false", pred_if_false)

        # black
        pred_true_loss = torch.mean(criterian_each(logits_black, label_true), 1)
        pred_false_loss = torch.mean(criterian_each(logits_black, label_false), 1)
        minus = torch.sub(pred_true_loss, pred_false_loss)
        minus_relu = minus_relu + active_relu(minus)
        # pred_if_false = torch.add(pred_if_false, active(torch.mul(minus, 10000)))
        # print("-->pred_if_false", pred_if_false)

        # white
        pred_true_loss = torch.mean(criterian_each(logits_white, label_true), 1)
        pred_false_loss = torch.mean(criterian_each(logits_white, label_false), 1)
        minus = torch.sub(pred_true_loss, pred_false_loss)
        minus_relu = minus_relu + active_relu(minus)
        # pred_if_false = torch.add(pred_if_false, active(torch.mul(minus, 10000)))
        # print("-->pred_if_false", pred_if_false)

        # psychiatric_or_mental_illness
        pred_true_loss = torch.mean(criterian_each(logits_illness, label_true), 1)
        pred_false_loss = torch.mean(criterian_each(logits_illness, label_false), 1)
        minus = torch.sub(pred_true_loss, pred_false_loss)
        minus_relu = minus_relu + active_relu(minus)
        label_05 = torch.tensor([0.5 for l in orig_labels], device=device).float()
        bias_loss = torch.mean(torch.sub(active(minus_relu).float(), label_05).float())

        loss = torch.add(acc_loss, bias_loss)
        # loss = bias_loss
        # loss = acc_loss

        return loss, acc_loss, bias_loss


    def loss_function_individual_subgroup(self, model, dataset_identity, tokenizer, padding, max_seq_length, predict_labels, base_l):

        all_logits = []
        # all_logits = torch.FloatTensor([])
        for i in range(len(identities)):
            identity = identities[i]
            print("-->identity", identity)

            datas = {'text': dataset_identity[identity], 'label': dataset_identity['label']}
            dataset = DataFrame(datas)

            logits, labels = self.get_predictions(model=model, dataset=dataset, tokenizer=tokenizer,
                                                  padding=padding,
                                                  max_seq_length=max_seq_length, if_identity=False)
            all_logits.append(logits)
            # all_logits += logits

        sum_logits = all_logits[0]
        for i in range(1, len(identities)):
            sum_logits = torch.add(sum_logits, all_logits[i])
        mean_logits = torch.mean(sum_logits)
        print("-->mean_logits", mean_logits)
        print("-->last logits", logits)
        # return logits

        criterion = nn.MSELoss()
        base_l = base_l.float()
        acc_loss = criterion(predict_labels, base_l)
        # bias_loss = torch.mean(torch.abs(x1 - x2))

        bias_loss = logits
        loss = bias_loss
        return loss, bias_loss, acc_loss

    def loss_function_test(self, predict_y, label_y):
        # print("-->(predict_y - label_y)", (predict_y - label_y))
        # print("-->torch.pow((predict_y - label_y), 2)", torch.pow(torch.tensor((predict_y - label_y)), 2))
        return torch.mean(torch.pow((predict_y - label_y), 2))

    def dataset_process(self, dataset):
        orig_text = []
        idi_list = [[], [], [], [], []]
        for i in range(0, len(dataset['orig_text'])):
            text = dataset['orig_text'].values.tolist()[i]
            orig_text.append(text)
            idis = eval(dataset['idis'].values.tolist()[i])
            for j in range(0, len(idi_list)):
                try:
                    idi = idis[j]
                except:
                    idi = text
                idi_list[j].append(idi)
        datas = {'orig_text': orig_text, 'idi1': idi_list[0], 'idi2': idi_list[1], 'idi3': idi_list[2], 'idi4': idi_list[3],
                 'idi5': idi_list[4], 'label':dataset['label']}
        processed_dataset = DataFrame(datas)
        return processed_dataset



    def debias_optimize(self, debias_dataset, tokenizer, padding, max_seq_length):

        # print("-->debias_dataset", debias_dataset)
        # print(type(debias_dataset))
        # debias_dataset = debias_dataset.iloc[0: 256]

        model = self.CustomModel.custom_model["custom_module"]

        if self.metric == None:
            # criterion = MyLoss()
            # 4.定义迭代优化算法， 使用的是随机梯度下降算法
            # todo: custom_model have parameters variable?
            # print("-->parameters")
            # weight = self.CustomModel.custom_model["custom_module"].linear.weight
            # print(weight)
            # bias = self.CustomModel.custom_model["custom_module"].linear.bias
            # print(bias)

            # print("-->original grad")
            # for name, parameters in model.named_parameters():
            #     print("grad:", name, ':', parameters.grad)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)  # SGD, Adam
            loss_dict = []
            bias_loss_dict = []
            acc_loss_dict = []

            labels = torch.tensor(debias_dataset['label'].values.tolist())[:10]
            labels.to(device)

            labels = torch.tensor([[(l+1)%2, (l+2)%2] for l in labels], device=device)
            labels.to(device)

            # Train the model 5. 迭代训练
            for epoch in range(self.num_epochs):
                print("-->epoch", epoch)
                # metric_score = self.metric_calculation(debias_dataset, tokenizer, padding, max_seq_length)
                # loss = criterion(metric_score)

                # 打乱debias_dataset的数据
                # permutation = np.random.permutation(len(self.y_train))

                loops = len(debias_dataset.index) // self.batch_size
                print("-->loops", loops)
                for i in range(0, loops):
                    print("-->i", i)
                    debias_dataset_one_batch = debias_dataset.iloc[i*self.batch_size: (i+1)*self.batch_size]
                    labels = torch.tensor(debias_dataset_one_batch['label'].values.tolist())
                    labels = torch.tensor([[(l + 1) % 2, (l + 2) % 2] for l in labels], device=device)
                    labels.to(device)

                    ## test with 1) get_Male Female prediction results and put results in loss function
                    # predicts_M, predicts_F = self.get_predictions(model, debias_dataset, tokenizer, padding, max_seq_length)
                    logits, labels = self.get_predictions(model, debias_dataset_one_batch, tokenizer, padding, max_seq_length, False)
                    print("-->get_predictions logits", logits)

                    predicts_M, predicts_F, logits_M, logits_F = self.get_predictions_all(model,
                                                                                          debias_dataset_one_batch,
                                                                                          tokenizer, padding,
                                                                                          max_seq_length)
                    # loss = self.loss_funtion_diff(predicts_M, predicts_F)
                    # print("-->logits_M", logits_M)
                    loss, bias_loss, acc_loss = self.loss_function_with_acc(predicts_M, predicts_F, logits_M, logits_F,
                                                                           labels)

                    # Backward and optimize 5.4 反向传播更新参数
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                    # new_weight = model.linear.weight
                    # new_bias = model.linear.bias
                    # new_weight = model.weight
                    # new_bias = model.bias
                    # if weight.equal(new_weight) and bias.equal(new_bias):
                    #     print("equal!")

                # 可选 5.5 打印训练信息和保存loss
                loss_dict.append(loss.item())
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.num_epochs, loss.item()))
                bias_loss_dict.append(bias_loss.item())
                acc_loss_dict.append(acc_loss.item())
                # if (epoch + 1) % 5 == 0:
                #     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

            # 画loss在迭代过程中的变化情况
            plt.plot(loss_dict, label='loss for every epoch')
            plt.plot(acc_loss_dict, label='mse loss')
            plt.legend()
            plt.savefig("loss" + str(self.num_epochs) + "_" + str(self.learning_rate) + ".png")
            plt.close()

            plt.plot(bias_loss_dict, label='bias loss')
            plt.savefig("loss_bias" + str(self.num_epochs) + "_" + str(self.learning_rate) + ".png")
            plt.close()

    def adjust_batch_size(self, model, classifier, tokenizer, padding, max_seq_length, dataset_identity, start_index):
        # training on data with identity
        # dataset_identity_one_batch = dataset_identity.iloc[i * self.batch_size: (i + 1) * self.batch_size]
        batch_size = self.MIN_BATCH_SIZE
        p0 = self.threshold - self.sigma
        p1 = self.threshold + self.sigma
        p0_ = 1 - p0
        p1_ = 1 - p1
        accept_bound = (1 - self.BETA) / self.ALPHA
        deny_bound = self.BETA / (1 - self.ALPHA)

        length = dataset_identity.shape[0]

        if start_index + self.MIN_BATCH_SIZE >= length:
            batch_size = length - start_index - 1
            return batch_size

        dataset_identity_one_batch = dataset_identity.iloc[start_index: start_index + batch_size]

        dataset_identity_one_batch = self.dataset_process(dataset_identity_one_batch)

        # logits, predict_labels = self.get_predictions(model, dataset_one_batch, tokenizer, padding, max_seq_length, False)
        all_logits, all_labels = self.get_predictions_all_idi(model, classifier, dataset_identity_one_batch, tokenizer,
                                                                   padding, max_seq_length)

        sum = np.array(all_labels[0]) + np.array(all_labels[1]) + np.array(all_labels[2]) + np.array(
            all_labels[3]) + np.array(all_labels[4]) + np.array(all_labels[5])
        sum = sum.tolist()
        ns = 0
        for j in sum:
            if j / 6.0 != 0 and j / 6.0 != 1:
                ns += 1
        s = batch_size - ns
        sprt_ratio = ((p1 / p0) ** s) * ((p1_ / p0_) ** ns)

        if sprt_ratio <= deny_bound:
            return batch_size
        else:
            start_index = start_index + batch_size
            batch_size += self.STEP

        while batch_size <= self.MAX_BATCH_SIZE:
            dataset_identity_added = dataset_identity.iloc[start_index: start_index + batch_size]
            # dataset_identity_added = dataset_identity.iloc[start_index: start_index + batch_size]
            dataset_identity_added = self.dataset_process(dataset_identity_added)

            all_logits, all_labels = self.get_predictions_all_idi(model, classifier, dataset_identity_added, tokenizer,
                                                                       padding, max_seq_length)

            sum = np.array(all_labels[0]) + np.array(all_labels[1]) + np.array(all_labels[2]) + np.array(
                all_labels[3]) + np.array(all_labels[4]) + np.array(all_labels[5])
            sum = sum.tolist()
            # print("-->all_logits", all_logits)
            # print("-->all_labels", all_labels)
            # print("-->sum", sum)
            # ns = 0
            for j in sum:
                if j / 6.0 != 0 and j / 6.0 != 1:
                    ns += 1
            s = batch_size - ns

            sprt_ratio = ((p1 / p0) ** s) * ((p1_ / p0_) ** ns)
            print("-->sprt_ratio:{} with ns:{}, s:{}".format(sprt_ratio, ns, s))

            if sprt_ratio > deny_bound and sprt_ratio < accept_bound and start_index + batch_size + self.STEP < length and \
                    batch_size + self.STEP <= self.MAX_BATCH_SIZE:
                batch_size += self.STEP
                start_index = start_index + self.STEP
                continue
            else:
                return batch_size

    def adjust_batch_size_fnr(self, model, classifier, tokenizer, padding, max_seq_length, dataset_identity, start_index):
        # training on data with identity
        # dataset_identity_one_batch = dataset_identity.iloc[i * self.batch_size: (i + 1) * self.batch_size]
        batch_size = self.MIN_BATCH_SIZE
        p0 = self.threshold - self.sigma
        p1 = self.threshold + self.sigma
        p0_ = 1 - p0
        p1_ = 1 - p1
        accept_bound = (1 - self.BETA) / self.ALPHA
        deny_bound = self.BETA / (1 - self.ALPHA)
        ID = IdentityDetect()

        length = dataset_identity.shape[0]

        if start_index + self.MIN_BATCH_SIZE >= length:
            batch_size = length - start_index - 1
            return batch_size

        # end_index = min(LENGTH, start_index + batch_size)
        dataset_identity_one_batch = dataset_identity.iloc[start_index: start_index + batch_size]

        all_logits, all_labels, overall_logits, overall_labels, all_orig_labels = \
            self.get_predictions_all_identity(model, classifier, dataset_identity_one_batch, tokenizer,
                                              padding, max_seq_length, ID)
        # all_logits, all_labels = self.get_predictions_all_identity(model, classifier, dataset_identity_one_batch, tokenizer,
        #                                                            padding, max_seq_length)
        P = sum(dataset_identity_one_batch['label'].values.tolist())
        N_s = 0
        all_weighted_FN = 0
        for i in range(0, len(all_logits)):
            logits = all_logits[i]
            labels = all_labels[i]
            if logits == None:
                continue
            orig_labels = all_orig_labels[i]
            if sum(orig_labels) == 0:
                continue
            weight = P/float(sum(orig_labels))
            FN = 0
            for j in range(0, len(labels)):
                pred_label = labels[j]
                orig_label = orig_labels[j]
                if pred_label == 0 and orig_label == 1:
                    FN += 1
            N_s += 1
            weighted_FN = FN * weight
            all_weighted_FN += weighted_FN

        n = P
        average_FN_weighted = all_weighted_FN / float(N_s)
        s = n - average_FN_weighted
        ns = average_FN_weighted
        print("-->n:{}, s:{}, ns: {}".format(n, s, ns))

        sprt_ratio = ((p1 / p0) ** s) * ((p1_ / p0_) ** ns)
        if sprt_ratio <= deny_bound:
            return batch_size

        while batch_size <= self.MAX_BATCH_SIZE:
            # dataset_identity_added = dataset_identity.iloc[start_index + batch_size: start_index + batch_size + self.STEP]
            dataset_identity_added = dataset_identity.iloc[start_index: start_index + batch_size + self.STEP]

            all_logits, all_labels, overall_logits, overall_labels, all_orig_labels = \
                self.get_predictions_all_identity(model, classifier, dataset_identity_added, tokenizer,
                                                  padding, max_seq_length, ID)
            # all_logits, all_labels = self.get_predictions_all_identity(model, dataset_identity_added, tokenizer,
            #                                                            padding, max_seq_length)
            P = sum(dataset_identity_added['label'].values.tolist())
            N_s = 0
            all_weighted_FN = 0
            for i in range(0, len(all_logits)):
                logits = all_logits[i]
                labels = all_labels[i]
                if logits == None:
                    continue

                orig_labels = all_orig_labels[i]
                if sum(orig_labels) == 0:
                    continue
                weight = P / float(sum(orig_labels))
                FN = 0
                for j in range(0, len(labels)):
                    pred_label = labels[j]
                    orig_label = orig_labels[j]
                    if pred_label == 0 and orig_label == 1:
                        FN += 1
                N_s += 1
                weighted_FN = FN * weight
                all_weighted_FN += weighted_FN


            n = P  # n + P
            average_FN_weighted = all_weighted_FN / float(N_s)
            s = n - average_FN_weighted   # s + n - average_FN_weighted
            ns = average_FN_weighted    # ns + average_FN_weighted

            print("-->n:{}, s:{}, ns: {}".format(n, s, ns))

            sprt_ratio = ((p1 / p0) ** s) * ((p1_ / p0_) ** ns)
            print("-->sprt_ratio:{} with ns:{}, s:{}".format(sprt_ratio, ns, s))

            if sprt_ratio > deny_bound and start_index + batch_size + self.STEP < length and \
                    batch_size + self.STEP <= self.MAX_BATCH_SIZE:
                batch_size += self.STEP
                continue
            else:
                return batch_size


    def find_idi(self, dataset_identity, tokenizer, padding, max_seq_length):
        all_labels = []
        for i in range(len(identities)):
            identity = identities[i]
            print("-->identity", identity)

            datas = {'text': dataset_identity[identity], 'label': dataset_identity['label']}
            dataset = DataFrame(datas)

            logits, labels = self.get_predictions_single(model=None, classifier=None, dataset=dataset, tokenizer=tokenizer,
                                                    padding=padding, max_seq_length=max_seq_length, if_identity=False)
            all_labels.append(labels)

        text = []
        label = []
        for j in range(0, len(dataset_identity)):
            label_base = all_labels[0][j]
            for i in range(1, len(identities)):
                identity = identities[i]
                label_identity = all_labels[i][j]
                if label_identity != label_base:
                    text.append(dataset_identity[identity].tolist()[j])
                    print("-->label_identity", label_identity)
                    print("-->label_base", label_base)
                    label.append(label_base)
        datas = {'text': text, 'label': label}
        dataset = DataFrame(datas)
        print("-->dataset", dataset)
        return dataset

    def debias_retraining(self, dataset, tokenizer, padding, max_seq_length):
        dataset = pd.read_csv("dataset/dataset.csv")
        dataset_identity = pd.read_csv("dataset/dataset_identity1.csv")
        dataset_idi = pd.read_csv("dataset/dataset_idi.csv")

        dataset_new = dataset.append(dataset_idi)

        # state_dict = torch.load('added_model_adapter.pth')  # added_model_adapter
        # print("-->model keys:", state_dict.keys())
        class AddedLayers(nn.Module):
            def __init__(self, n_feature, hidden_output, n_output):
                super(AddedLayers, self).__init__()

                self.model = nn.Sequential(
                    nn.Linear(in_features=n_feature, out_features=hidden_output),
                    nn.GELU(),
                    nn.Linear(in_features=hidden_output, out_features=n_output),

                    nn.ReLU(),
                    nn.Linear(in_features=n_output, out_features=hidden_output),
                    nn.ReLU(),
                    nn.Linear(in_features=hidden_output, out_features=n_output),
                )

            def forward(self, input):
                w1 = self.model[0].weight.t()
                b1 = self.model[0].bias
                net = torch.tensordot(input, w1, [[1], [0]]) + b1
                net = self.model[1](net)
                w2 = self.model[2].weight.t()
                b2 = self.model[2].bias
                output = torch.tensordot(net, w2, [[1], [0]]) + b2

                net = self.model[3](net)
                w2 = self.model[4].weight.t()
                b2 = self.model[4].bias
                output = torch.tensordot(net, w2, [[1], [0]]) + b2

                net = self.model[5](output)
                w2 = self.model[6].weight.t()
                b2 = self.model[6].bias
                output = torch.tensordot(net, w2, [[1], [0]]) + b2

                return output

        model = AddedLayers(768, 61, 768)
        model.to(device)
        # model.load_state_dict(state_dict)

        # print("-->original metric score:")
        # metric = self.get_metrics_single(model, dataset_identity_test, tokenizer, padding, max_seq_length, 'fnr')
        # print("-->fnr metric", metric)
        # metric = self.get_metrics_single(model, dataset_identity_test, tokenizer, padding, max_seq_length, 'individual')
        # print("-->individual metric", metric)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)  # SGD, Adam
        loss_dict = []
        bias_loss_dict = []
        acc_loss_dict = []

        orig_weight = model.model[0].weight.clone()
        print("-->orig_weight", orig_weight)
        orig_bias = model.model[0].bias.clone()
        print("-->orig_bias", orig_bias)

        # First train the model on acc_loss only
        batch_size = 512
        for epoch in range(10):
            print("-->epoch", epoch)
            loops = len(dataset_new.index) // batch_size
            print("-->loops", loops)
            for i in range(0, loops):
                print("-->i", i)
                dataset_one_batch = dataset_new.iloc[i * batch_size: (i + 1) * batch_size]
                orig_labels = torch.tensor(dataset_one_batch['label'].values.tolist(), device=device)
                orig_labels = orig_labels.float()
                labels = torch.tensor([[(l + 1) % 2, (l + 2) % 2] for l in orig_labels], device=device)
                labels = labels.float()
                labels.to(device)

                logits, predict_labels = self.get_predictions(model, dataset_one_batch, tokenizer, padding,
                                                              max_seq_length, False)

                loss = self.loss_function_MSE(labels, logits)
                print("-->loss", loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # for name, parameters in model.named_parameters():
                #     print("grad:", name, ':', parameters.grad)

                # 打印训练信息和保存loss
                loss_dict.append(loss.item())
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.num_epochs, loss.item()))

        # torch.save(model.model, 'added_model2_mse.pkl')
        torch.save(model.state_dict(), 'added_model_retrianing.pth')

        print("-->after metric score:")
        metric = self.get_metrics_single(model, dataset_identity, tokenizer, padding, max_seq_length, 'fnr')
        print("-->fnr metric", metric)
        metric = self.get_metrics_single(model, dataset_identity, tokenizer, padding, max_seq_length, 'individual')
        print("-->individual metric", metric)

        print("-->get baseline FNR after fine-tuning ")
        logtis, predict_labels = self.get_predictions_single(model, dataset, tokenizer, padding, max_seq_length,
                                                             False)
        TP, FP, TN, FN = self.perf_measure(dataset['label'].tolist(), predict_labels)
        print("-->FPR:", FP / float(FP + TN))
        print("-->FNR:", FN / float(FN + TP))

    def get_BCEWithLogitsLoss(self, dataset):
        text = dataset['text']
        orig_labels = dataset_identity_one_batch['label'].values.tolist()

    def filter_unprivileged_dataset(self, dataset_identity, start_index):
        texts = dataset_identity['orig_text'].values.tolist()
        labels = dataset_identity['label'].values.tolist()
        # idis = dataset_identity['idis'].values.tolsit()
        idis = dataset_identity['idis']
        new_orig_texts = []
        new_idis = []
        new_labels = []
        for i in range(start_index, len(dataset_identity)):
            text = texts[i]
            label = labels[i]
            idi = idis[i]
            if label == self.privileged_label:
                new_orig_texts.append(text)
                new_labels.append(label)
                new_idis.append(idi)
            if len(new_orig_texts) >= self.batch_size or i == len(dataset_identity) - 1:
                datas = {"orig_text": new_orig_texts, "idis": new_idis, "label": new_labels}
                dataset = pd.DataFrame(datas)
                end_index = i + 1
                # print("-->dataset", dataset)
                # print("-->end_index", end_index)
                return dataset, end_index

    def debias_optimize_identity(self, tokenizer, padding, max_seq_length, baseline_metric_ind, baseline_metric_group, baseline_fpr_train, baseline_fnr_train, save_dir):
        # dataset = pd.read_csv("dataset/dataset.csv")
        # dataset_identity = pd.read_csv("dataset/dataset_identity1.csv")
        # dataset_identity = pd.read_csv("dataset/dataset_identity.csv")
        # dataset_identity_test = pd.read_csv("dataset/dataset_identity_test.csv")

        dataset_train = self.train_data.sample(frac=1, random_state=999)
        dataset_test = self.test_data.sample(frac=1, random_state=999)
        dataset_identity = self.train_data_identity.sample(frac=1, random_state=999)
        dataset_identity_test = self.test_data_identity.sample(frac=1, random_state=999)

        # model = self.CustomModel.custom_model["custom_module"]

        model = self.CustomModel.custom_model['custom_module']
        # print("-->model", model)
        print("-->model.parameters()", model.parameters())
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)  # SGD, Adam
        orig_classifier = copy.copy(self.CustomModel.custom_model['classifier'])
        classifier = self.CustomModel.custom_model['classifier']
        # print("-->classifier", classifier)
        print("-->classifier", classifier.parameters())
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': self.learning_rate},
                                     {'params': classifier.parameters(), 'lr': self.learning_rate}])

        # default
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        loss_dict = []
        bias_loss_dict = []
        acc_loss_dict = []
        ind_metrics_train = []
        ind_metrics_test = []
        group_metrics_train = []
        group_metrics_test = []
        overall_acc_train = []
        overall_acc_test = []

        orig_weight = model.model[0].weight.clone()
        print("-->orig_weight", orig_weight)
        orig_bias = model.model[0].bias.clone()
        print("-->orig_bias", orig_bias)

        classifier_orig_weight = classifier.weight.clone()
        # model = torch.load('added_model.pkl')

        # print("training datas")
        # print("-->original data metric:")
        # metric = self.get_metrics_single(None, None, dataset_identity, tokenizer, padding, max_seq_length,
        #                                  'data metric')  # dataset_identity_test
        # print("data metric", metric)
        # #
        # print("testing datas")
        # print("-->original data score:")
        # metric = self.get_metrics_single(None, None, dataset_identity_test, tokenizer, padding, max_seq_length,
        #                                  'data metric')  # dataset_identity_test
        # print("data metric", metric)
        #
        # print("-->overall baseline on training dataset")
        # logtis, predict_labels = self.get_predictions_single(None, None, dataset_train, tokenizer, padding,
        #                                                      max_seq_length,
        #                                                      False)
        # label = dataset_train['label'].tolist()
        # TP, FP, TN, FN = self.perf_measure(label, predict_labels)
        # from sklearn.metrics import accuracy_score, f1_score
        # print("-->ACC", accuracy_score(label, predict_labels))
        # print("-->F1", f1_score(label, predict_labels))
        # print("-->FPR:", FP / float(FP + TN))
        # print("-->FNR:", FN / float(FN + TP))
        #
        # print("-->overall baseline on testing dataset")
        # logtis, predict_labels = self.get_predictions_single(None, None, dataset_test, tokenizer, padding,
        #                                                      max_seq_length,
        #                                                      False)
        # label = dataset_test['label'].tolist()
        # TP, FP, TN, FN = self.perf_measure(label, predict_labels)
        # from sklearn.metrics import accuracy_score, f1_score
        # print("-->ACC", accuracy_score(label, predict_labels))
        # print("-->F1", f1_score(label, predict_labels))
        # print("-->FPR:", FP / float(FP + TN))
        # print("-->FNR:", FN / float(FN + TP))

        # print("training dataset")
        # print("-->original metric score:")
        # metric = self.get_metrics_single(None, None, dataset_identity, tokenizer, padding, max_seq_length,
        #                                  'individual')  # dataset_identity_test
        # print("-->individual metric", metric)
        # baseline_metric_ind = metric
        metric, fnr_metric, fpr, fnr = self.get_metrics_single(None, None, dataset_identity, tokenizer, padding, max_seq_length, 'fnr',  baseline_fpr_train, baseline_fnr_train)
        print("-->mean fpr diff metric", metric)

        # print("testing dataset")
        # print("-->original metric score:")
        # metric = self.get_metrics_single(None, None, dataset_identity_test, tokenizer, padding, max_seq_length,
        #                                  'individual')  # dataset_identity_test
        # print("-->individual metric", metric)
        metric, fnr_metric, fpr, fnr = self.get_metrics_single(None, None, dataset_identity_test, tokenizer, padding, max_seq_length, 'fnr',  baseline_fpr_train, baseline_fnr_train)
        print("-->mean fpr diff metric", metric)

        return


        if self.target == "individual":
            min_p = 0
            max_p = baseline_metric_ind + 0.1
        elif self.target == "group":
            min_p = 0
            # max_p = 0.2
            max_p = baseline_metric_group + 0.1
        else:
            print("self.target ERROR")
            return

        # max_p = 0.5
        MinValue = (math.sqrt(self.batch_size) * (min_p - self.threshold)) / \
                   (math.sqrt(self.threshold * (1 - self.threshold)))
        MaxValue = (math.sqrt(self.batch_size) * (max_p - self.threshold)) / \
                   (math.sqrt(self.threshold * (1 - self.threshold)))

        print("-->MinValue:{}, MaxValue:{}".format(MinValue, MaxValue))

        train_ind = []
        test_ind = []
        train_acc = []
        train_f1 = []
        train_fpr = []
        train_fnr = []
        test_acc = []
        test_f1 = []
        test_fpr = []
        test_fnr = []

        # Train the model 迭代训练
        for epoch in range(self.num_epochs):
            print("-->epoch", epoch)
            start_index = 0
            end_index = 0
            while end_index < len(dataset_identity) - 1:
                print("-->start_index", start_index)
                batch_size = self.batch_size
                print("-->batch_size", batch_size)
                if self.target == "individual":
                    end_index = min(len(dataset_identity) - 1, start_index + batch_size)
                    dataset_identity_one_batch = dataset_identity.iloc[start_index: end_index]
                    orig_labels = dataset_identity_one_batch['label'].values.tolist()
                elif self.target == "group":
                    dataset_identity_one_batch, output_end_index = self.filter_unprivileged_dataset(dataset_identity, start_index)
                    end_index = output_end_index
                    dataset_one_batch = dataset_identity.iloc[start_index: end_index]
                    orig_labels = dataset_one_batch['label'].values.tolist()

                # labels = torch.tensor([[(l + 1) % 2, (l + 2) % 2] for l in orig_labels], device=device)
                # labels = labels.float()
                # labels.to(device)
                #
                # label_0 = torch.tensor([[1, 0] for l in orig_labels], device=device).float()
                # label_1 = torch.tensor([[0, 1] for l in orig_labels], device=device).float()
                # label_0.to(device)
                # label_1.to(device)

                dataset_identity_one_batch = self.dataset_process(dataset_identity_one_batch)

                if self.target == "individual":
                    # individual bias mitigation
                    all_logits, all_labels = self.get_predictions_all_idi(model, classifier, dataset_identity_one_batch,
                                                                          tokenizer,
                                                                          padding, max_seq_length)
                    orig_logits = self.get_predictions_all_basic(orig_classifier, dataset_identity_one_batch, tokenizer, padding, max_seq_length)

                    loss, acc_loss, bias_loss = self.loss_function_sprt_individual(orig_labels, all_logits, all_labels, orig_logits, MinValue, MaxValue, self.gamma)

                else:
                    # group bias mitigation
                    ID = IdentityDetect()
                    all_logits, all_labels, orig_all_labels = \
                        self.get_predictions_all_identity(model, classifier, dataset_identity_one_batch, tokenizer,
                                                          padding, max_seq_length, ID)
                    predict_logits, predict_labels = self.get_predictions_all(model, classifier, dataset_one_batch,
                                                                              tokenizer,
                                                                              padding, max_seq_length)
                    # P_overall = torch.sum(orig_labels)
                    identity_num = [len(labels) for labels in orig_all_labels]
                    # print("-->identity_num", identity_num)
                    show_identity_num = sum(1 for P in identity_num if P > 0)
                    # print("-->show_identity_num", show_identity_num)
                    # print("-->orig_labels", orig_labels)
                    # print("-->overall_logits", overall_logits)
                    # print("-->all_logits", all_logits)
                    # print("-->identity_num", identity_num)
                    loss, acc_loss, bias_loss = self.loss_function_sprt_fpr(orig_labels, predict_logits, identity_num, show_identity_num,
                                                                            MinValue, MaxValue, all_logits, baseline_fpr_train)

                optimizer.zero_grad()
                loss.backward()
                # Apply gradient clipping separately to the editor and classifier
                # max_grad_norm = 5.0
                # clip_grad_norm_(model.parameters(), max_grad_norm)
                # clip_grad_norm_(classifier.parameters(), max_grad_norm)
                optimizer.step()

                # 打印训练信息和保存loss
                loss_dict.append(loss.item())
                print(
                    'Epoch [{}/{}], Loss: {:.4f}, acc_loss, {:.4f}, bias_loss:{:.4f}'.format(epoch + 1, self.num_epochs,
                                                                                             loss.item(),
                                                                                             acc_loss.item(),
                                                                                             bias_loss.item()))
                bias_loss_dict.append(bias_loss)
                acc_loss_dict.append(acc_loss.item())

                # new_weight = model.weight.clone()
                # new_bias = model.bias.clone()
                new_weight = model.model[0].weight.clone()
                new_bias = model.model[0].bias.clone()
                if torch.equal(new_weight, orig_weight) == True:
                    print("model weight equal.")

                classifier_new_weight = classifier.weight.clone()
                if torch.equal(classifier_new_weight, classifier_orig_weight) == True:
                    print("classifier weight equal.")

                start_index = end_index

            print("-->after epoch", epoch)
            print("training dataset")
            print("-->original metric score:")
            training_indi_metric = self.get_metrics_single(model, classifier, dataset_identity, tokenizer, padding, max_seq_length,
                                             'individual')  # dataset_identity_test
            print("-->individual metric", training_indi_metric)
            ind_metrics_train.append(training_indi_metric)
            train_ind.append(training_indi_metric)

            # training_group_metirc = self.get_metrics_single(model, classifier, dataset_identity, tokenizer, padding, max_seq_length, 'fnr')
            training_group_metirc, fnr_metric, fpr, fnr  = self.get_metrics_single(model, classifier, dataset_identity, tokenizer, padding, max_seq_length,
                                             'fpr', baseline_fpr_train, baseline_fnr_train)
            print("-->mean fpr diff metric", training_group_metirc)
            group_metrics_train.append(training_group_metirc)

            print("testing dataset")
            print("-->original metric score:")
            metric = self.get_metrics_single(model, classifier, dataset_identity_test, tokenizer, padding, max_seq_length,
                                             'individual')  # dataset_identity_test
            print("-->individual metric", metric)
            ind_metrics_test.append(metric)
            test_ind.append(metric)

            # metric = self.get_metrics_single(model, classifier, dataset_identity_test, tokenizer, padding, max_seq_length, 'fnr')
            metric, fnr_metric, fpr, fnr = self.get_metrics_single(model, classifier, dataset_identity_test, tokenizer, padding, max_seq_length,
                                             'fpr', baseline_fpr_train, baseline_fnr_train)
            print("-->mean fnr diff metric", metric)
            group_metrics_test.append(metric)

            print("-->overall baseline on training dataset")
            logtis, predict_labels = self.get_predictions_single(model, classifier, dataset_train, tokenizer, padding, max_seq_length,
                                                                 False)
            label = dataset_train['label'].tolist()
            TP, FP, TN, FN = self.perf_measure(label, predict_labels)
            from sklearn.metrics import accuracy_score, f1_score
            training_acc = accuracy_score(label, predict_labels)
            f1 = f1_score(label, predict_labels)
            fpr = FP / float(FP + TN)
            fnr = FN / float(FN + TP)
            print("-->ACC", training_acc)
            print("-->F1", f1)
            print("-->FPR:", fpr)
            print("-->FNR:", fnr)
            overall_acc_train.append(training_acc)
            train_acc.append(training_acc)
            train_f1.append(f1)
            train_fpr.append(fpr)
            train_fnr.append(fnr)

            print("-->overall baseline on testing dataset")
            logtis, predict_labels = self.get_predictions_single(model, classifier, dataset_test, tokenizer, padding, max_seq_length,
                                                                 False)
            label = dataset_test['label'].tolist()
            TP, FP, TN, FN = self.perf_measure(label, predict_labels)
            from sklearn.metrics import accuracy_score, f1_score
            testing_acc = accuracy_score(label, predict_labels)
            f1 = f1_score(label, predict_labels)
            fpr = FP / float(FP + TN)
            fnr = FN / float(FN + TP)
            print("-->ACC", testing_acc)
            print("-->F1", f1_score(label, predict_labels))
            print("-->FPR:", FP / float(FP + TN))
            print("-->FNR:", FN / float(FN + TP))
            overall_acc_test.append(testing_acc)
            test_acc.append(testing_acc)
            test_f1.append(f1)
            test_fpr.append(fpr)
            test_fnr.append(fnr)

            path_directory = save_dir + str(self.target) + "_" + str(self.orig_threshold) + \
                             "_" + str(self.learning_rate) + "_" + str(self.gamma) + "_group" + "/"
            print("-->path_directory", path_directory)
            os.makedirs(path_directory, exist_ok=True)
            model_path = path_directory + 'editor' + str(epoch) + '.pth'
            torch.save(model.state_dict(), model_path)
            model_path = path_directory + 'classifier' + str(epoch) + '.pth'
            torch.save(classifier.state_dict(), model_path)

            if self.target == "individual":
                if training_indi_metric < self.threshold and training_acc >= 0.8:
                    # self.threshold -= 0.005
                    print("-->threshold", self.threshold)
                    # self.batch_size = int(((1.65 + 0.84) / 0.05)**2 * self.threshold * (1 - self.threshold))
                    # print("-->batch_size", self.batch_size)
                    #n=(POWER((C16+D16)/E16,2))*(F16*(1-F16))
            else:
                if training_group_metirc < self.threshold and training_acc >= 0.8:
                    # self.threshold -= 0.005
                    print("-->threshold", self.threshold)
                    # self.batch_size = int(((1.65 + 0.84) / 0.05) ** 2 * self.threshold * (1 - self.threshold))
                    # print("-->batch_size", self.batch_size)

            exp_lr_scheduler.step()


        figure_path_directory = path_directory + "figure/"
        os.makedirs(figure_path_directory, exist_ok=True)
        plt.plot(ind_metrics_train, label='training individual metrics')
        plt.legend()
        plt.savefig(figure_path_directory + "individual_metrics_train.png")
        plt.close()

        plt.plot(ind_metrics_test, label='testing individual metrics')
        plt.legend()
        plt.savefig(figure_path_directory + "individual_metrics_test.png")
        plt.close()

        plt.plot(group_metrics_train, label='training group metrics')
        plt.legend()
        plt.savefig(figure_path_directory + "group_metrics_train.png")
        plt.close()

        plt.plot(group_metrics_test, label='testing group metrics')
        plt.legend()
        plt.savefig(figure_path_directory + "group_metrics_test.png")
        plt.close()

        plt.plot(overall_acc_train, label='training acc')
        plt.legend()
        plt.savefig(figure_path_directory + "acc_train.png")
        plt.close()

        plt.plot(overall_acc_test, label='testing acc')
        plt.legend()
        plt.savefig(figure_path_directory + "acc_test.png")
        plt.close()

        datas = {"train_ind": train_ind, "train_acc": train_acc, "train_f1": train_f1, "train_fpr": train_fpr,
                 "train_fnr": train_fnr,
                 "test_ind": test_ind, "test_acc": test_acc, "test_f1": test_f1, "test_fpr": test_fpr,
                 "test_fnr": test_fnr}
        dataset = pd.DataFrame(datas)
        print(dataset)
        dataset.to_csv(path_directory + "figure/result.csv")

    def debias_optimize_identity_diff_gamma(self, tokenizer, padding, max_seq_length, baseline_metric_ind, baseline_metric_group, save_dir, gamma,
                                            baseline_fpr_train, baseline_fnr_train):
        print("-->gamma", gamma)

        dataset_train = self.train_data.sample(frac=1, random_state=999)
        dataset_test = self.test_data.sample(frac=1, random_state=999)
        dataset_identity = self.train_data_identity.sample(frac=1, random_state=999)
        dataset_identity_test = self.test_data_identity.sample(frac=1, random_state=999)

        # model = self.CustomModel.custom_model["custom_module"]

        model = self.CustomModel.custom_model['custom_module']
        # print("-->model", model)
        print("-->model.parameters()", model.parameters())
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)  # SGD, Adam
        orig_classifier = copy.copy(self.CustomModel.custom_model['classifier'])
        classifier = self.CustomModel.custom_model['classifier']
        # print("-->classifier", classifier)
        print("-->classifier", classifier.parameters())
        optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': self.learning_rate},
                                     {'params': classifier.parameters(), 'lr': self.learning_rate}])

        # default
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        loss_dict = []
        bias_loss_dict = []
        acc_loss_dict = []
        ind_metrics_train = []
        ind_metrics_test = []
        group_metrics_train = []
        group_metrics_test = []
        overall_acc_train = []
        overall_acc_test = []

        orig_weight = model.model[0].weight.clone()
        print("-->orig_weight", orig_weight)
        orig_bias = model.model[0].bias.clone()
        print("-->orig_bias", orig_bias)

        classifier_orig_weight = classifier.weight.clone()
        # model = torch.load('added_model.pkl')

        if self.target == "individual":
            min_p = 0
            max_p = baseline_metric_ind + 0.1
        elif self.target == "group":
            min_p = 0
            # max_p = 0.1
            max_p = baseline_metric_group + 0.1
            # max_p = baseline_fpr_train + 0.1
        elif self.target == "idi":
            min_p = 0
            max_p = baseline_metric_ind
        else:
            print("self.target ERROR")
            return

        # max_p = 0.5
        MinValue = (math.sqrt(self.batch_size) * (min_p - self.threshold)) / \
                   (math.sqrt(self.threshold * (1 - self.threshold)))
        MaxValue = (math.sqrt(self.batch_size) * (max_p - self.threshold)) / \
                   (math.sqrt(self.threshold * (1 - self.threshold)))

        print("-->MinValue:{}, MaxValue:{}".format(MinValue, MaxValue))

        # # original group metric
        # metric, fnr_metric, mean_fpr, mean_fnr, fpr, fnr = self.get_metrics_single(None, None, dataset_identity, tokenizer, padding,
        #                                                        max_seq_length, 'fnr', baseline_fpr_train,
        #                                                        baseline_fnr_train)
        # print("-->base group metric", metric, mean_fpr)
        #
        # metric, fnr_metric, mean_fpr, mean_fnr, fpr, fnr = self.get_metrics_single(None, None, dataset_identity_test, tokenizer, padding,
        #                                                        max_seq_length, 'fnr', baseline_fpr_train,
        #                                                        baseline_fnr_train)
        # print("-->base group metric", metric, mean_fpr)

        # training_indi_metric, gender_metric, religion_metric, race_metric = self.get_metrics_single(None, None,
        #                                                                                             dataset_identity_test,
        #                                                                                             tokenizer, padding,
        #                                                                                             max_seq_length,
        #                                                                                             'individual')  # dataset_identity_test
        # print("-->individual metric", training_indi_metric)

        train_ind = []
        test_ind = []
        train_acc = []
        train_f1 = []
        train_fpr = []
        train_fnr = []
        test_acc = []
        test_f1 = []
        test_fpr = []
        test_fnr = []

        train_gender_ind = []
        test_gender_ind = []
        train_religion_ind = []
        test_religion_ind = []
        train_race_ind = []
        test_race_ind = []

        if self.target == "group":
            train_fpr_all = []
            train_fpr_mean = []
            train_fpr_diff = []
            train_fnr_all = []
            train_fnr_mean = []
            train_fnr_diff = []

            test_fpr_all = []
            test_fpr_mean = []
            test_fpr_diff = []
            test_fnr_all = []
            test_fnr_mean = []
            test_fnr_diff = []

        train_fpr_mean = []
        train_fpr_diff = []
        train_fnr_mean = []
        train_fnr_diff = []

        test_fpr_mean = []
        test_fpr_diff = []
        test_fnr_mean = []
        test_fnr_diff = []

        # Train the model 迭代训练
        fpr_epoch_train = None
        fur_epoch_train = None

        if self.target == "idi":
            data_name = self.train_data_name.split(".")[0] + "_add_idis1.csv"
            # data_name = self.train_data_name.split(".")[0] + "_combine.csv"
            print("-->data_name", data_name)
            dataset_idi_combine = pd.read_csv(data_name)
            # dataset_idi_combine = dataset_train.copy()
            for epoch in range(self.num_epochs):
                print("-->epoch", epoch)
                start_index = 0
                end_index = 0
                while end_index < len(dataset_idi_combine) - 1:
                    # print("-->start_index", start_index)
                    batch_size = self.batch_size
                    # print("-->batch_size", batch_size)
                    end_index = min(len(dataset_idi_combine) - 1, start_index + batch_size)
                    dataset_one_batch = dataset_idi_combine.iloc[start_index: end_index]
                    # orig_labels = dataset_one_batch['label'].values.tolist()
                    # dataset_one_batch = self.dataset_process(dataset_one_batch)

                    all_logits, all_labels = self.get_predictions_all(model, classifier, dataset_one_batch,
                                                                          tokenizer,
                                                                          padding, max_seq_length)
                    criterian = nn.BCELoss(reduction='mean')
                    # criterian = nn.BCEWithLogitsLoss(reduction='mean')

                    # acc_loss
                    truth_labels = dataset_one_batch['label']
                    labels = torch.tensor([[(l + 1) % 2, (l + 2) % 2] for l in truth_labels], device=device)
                    labels = labels.float()
                    acc_loss = criterian(all_logits, labels)
                    loss = acc_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    start_index = end_index

            print("-->after epoch", epoch)

            print("-->overall baseline on training dataset")
            logtis, predict_labels = self.get_predictions_single(model, classifier, dataset_train, tokenizer,
                                                                 padding,
                                                                 max_seq_length,
                                                                 False)
            label = dataset_train['label'].tolist()
            TP, FP, TN, FN = self.perf_measure(label, predict_labels)
            from sklearn.metrics import accuracy_score, f1_score
            acc_train = accuracy_score(label, predict_labels)
            f1_train = f1_score(label, predict_labels)
            fpr_epoch_train = FP / float(FP + TN)
            fnr_epoch_train = FN / float(FN + TP)
            print("-->ACC", acc_train)
            print("-->F1", f1_train)
            print("-->FPR:", fpr_epoch_train)
            print("-->FNR:", fnr_epoch_train)
            train_acc.append(train_acc)
            train_f1.append(f1_train)
            train_fpr.append(fpr_epoch_train)
            train_fnr.append(fnr_epoch_train)

            print("-->overall baseline on testing dataset")
            logtis, predict_labels = self.get_predictions_single(model, classifier, dataset_test, tokenizer,
                                                                 padding,
                                                                 max_seq_length,
                                                                 False)
            label = dataset_test['label'].tolist()
            TP, FP, TN, FN = self.perf_measure(label, predict_labels)
            from sklearn.metrics import accuracy_score, f1_score
            acc_test = accuracy_score(label, predict_labels)
            f1_test = f1_score(label, predict_labels)
            fpr_epoch_test = FP / float(FP + TN)
            fnr_epoch_test = FN / float(FN + TP)
            print("-->ACC", acc_test)
            print("-->F1", f1_test)
            print("-->FPR:", fpr_epoch_test)
            print("-->FNR:", fnr_epoch_test)
            test_acc.append(acc_test)
            test_f1.append(f1_test)
            test_fpr.append(fpr_epoch_test)
            test_fnr.append(fnr_epoch_test)

            print("training dataset")
            print("-->original metric score:")
            train_ind_metric, gender_metric_train, religion_metric_train, race_metric_train = self.get_metrics_single(
                model,
                classifier,
                dataset_identity,
                tokenizer,
                padding,
                max_seq_length,
                'individual')  # dataset_identity_test
            print("-->individual metric", train_ind_metric)
            train_ind.append(train_ind_metric)
            train_gender_ind.append(gender_metric_train)
            train_religion_ind.append(religion_metric_train)
            train_race_ind.append(race_metric_train)

            # train_fpr_metirc, train_fnr_metric, train_mean_fpr, train_mean_fnr, train_fpr, train_fnr = \
            #     self.get_metrics_single(model, classifier, dataset_identity, tokenizer, padding, max_seq_length,
            #                             'fnr',
            #                             fpr_epoch_train, fnr_epoch_train)
            # print("-->mean fnr diff metric", train_fpr_metirc)
            # train_fpr_mean.append(train_mean_fpr)
            # train_fnr_mean.append(train_mean_fnr)
            # train_fpr_diff.append(train_fpr_metric)
            # train_fnr_diff.append(train_fnr_metric)

            print("testing dataset")
            print("-->original metric score:")
            test_ind_metric, gender_metric_test, religion_metric_test, race_metric_test = self.get_metrics_single(
                model, classifier,
                dataset_identity_test,
                tokenizer, padding,
                max_seq_length,
                'individual')  # dataset_identity_test
            print("-->individual metric", test_ind_metric)
            test_ind.append(test_ind_metric)
            test_gender_ind.append(gender_metric_test)
            test_religion_ind.append(religion_metric_test)
            test_race_ind.append(race_metric_test)

            # test_fpr_metric, test_fnr_metric, test_mean_fpr, test_mean_fnr, test_fpr, test_fnr = \
            #     self.get_metrics_single(model, classifier, dataset_identity_test, tokenizer, padding,
            #                             max_seq_length, 'fnr',
            #                             fpr_epoch_test, fnr_epoch_test)
            # print("-->mean fnr diff metric", test_fpr_metric)
            # test_fpr_mean.append(test_mean_fpr)
            # test_fnr_mean.append(test_mean_fnr)
            # test_fpr_diff.append(test_fpr_metric)
            # test_fnr_diff.append(test_fnr_metric)


            # datas = {'train_ind': train_ind, 'train_gender': train_gender_ind, 'train_religion': train_religion_ind,
            #          'train_race': train_race_ind, 'train_acc': train_acc, 'train_f1': train_f1,
            #          'train_fpr': train_fpr, 'train_fnr': train_fnr, 'train_fpr_mean': train_fpr_mean,
            #          'train_fpr_diff': train_fpr_diff,	'train_fnr_mean': train_fnr_mean, 'train_fnr_diff': train_fnr_diff,
            #          'test_ind': test_ind, 'test_gender': test_gender_ind,
            #          'test_religion': test_religion_ind,
            #          'test_race': test_race_ind, 'test_cc': test_acc, 'test_f1': test_f1,
            #          'test_fpr': test_fpr, 'test_fnr': test_fnr, 'test_fpr_mean': test_fpr_mean,
            #          'test_fpr_diff': test_fpr_diff, 'test_fnr_mean': test_fnr_mean,
            #          'test_fnr_diff': test_fnr_diff
            #          }
            datas = {'train_ind': train_ind, 'train_gender': train_gender_ind, 'train_religion': train_religion_ind,
                     'train_race': train_race_ind, 'train_acc': train_acc, 'train_f1': train_f1,
                     'train_fpr': train_fpr, 'train_fnr': train_fnr,
                     'test_ind': test_ind, 'test_gender': test_gender_ind,
                     'test_religion': test_religion_ind,
                     'test_race': test_race_ind, 'test_acc': test_acc, 'test_f1': test_f1,
                     'test_fpr': test_fpr, 'test_fnr': test_fnr
                     }
            dataset = pd.DataFrame(datas)

            path_directory = save_dir + str(self.target) + "/"
            os.makedirs(path_directory, exist_ok=True)

            model_path = path_directory + 'editor' + str(epoch) + '.pth'
            torch.save(model.state_dict(), model_path)
            model_path = path_directory + 'classifier' + str(epoch) + '.pth'
            torch.save(classifier.state_dict(), model_path)

            return dataset

        for epoch in range(self.num_epochs):
            print("-->epoch", epoch)
            start_index = 0
            end_index = 0
            while end_index < len(dataset_identity) - 1:
                print("-->start_index", start_index)
                batch_size = self.batch_size
                print("-->batch_size", batch_size)
                if self.target == "individual":
                    end_index = min(len(dataset_identity) - 1, start_index + batch_size)
                    dataset_identity_one_batch = dataset_identity.iloc[start_index: end_index]
                    orig_labels = dataset_identity_one_batch['label'].values.tolist()
                    dataset_identity_one_batch = self.dataset_process(dataset_identity_one_batch)
                elif self.target == "group":
                    # dataset_identity_one_batch: dataset with only privileged group_truth label
                    dataset_identity_one_batch, output_end_index = self.filter_unprivileged_dataset(dataset_identity,
                                                                                                    start_index)
                    end_index = output_end_index
                    # dataset_one_batch: dataset containing dataset_identity_one_batch with both 0 and 1 group-truth labels
                    dataset_one_batch = dataset_identity.iloc[start_index: end_index]
                    orig_labels = dataset_one_batch['label'].values.tolist()

                if self.target == "individual":
                    # individual bias mitigation
                    all_logits, all_labels = self.get_predictions_all_idi(model, classifier, dataset_identity_one_batch,
                                                                          tokenizer,
                                                                          padding, max_seq_length)
                    orig_logits = self.get_predictions_all_basic(orig_classifier, dataset_identity_one_batch, tokenizer, padding, max_seq_length)

                    loss, acc_loss, bias_loss = self.loss_function_sprt_individual(orig_labels, all_logits, all_labels, orig_logits, MinValue, MaxValue, gamma)

                elif self.target == "group":
                    # group bias mitigation
                    ID = IdentityDetect()
                    # prediction on dataset_identity_one_batch with each identity subgroup
                    # all_logits, all_labels, orig_all_labels = \
                    #     self.get_predictions_all_identity(model, classifier, dataset_identity_one_batch, tokenizer,
                    #                                       padding, max_seq_length, ID)
                    # prediction on dataset_one_batch only
                    # predict_logits, predict_labels = self.get_predictions_all(model, classifier, dataset_one_batch,
                    #                                                           tokenizer,
                    #                                                           padding, max_seq_length)
                    # prediction on dataset_one_batch and each identity subgroup
                    all_logits, all_labels, predict_logits, predict_labels, orig_all_labels = \
                        self.get_predictions_all_identity1(model, classifier, dataset_one_batch, tokenizer,
                                                           padding, max_seq_length, ID)

                    identity_num = [len(labels) for labels in orig_all_labels]
                    identity_privileged_num = [labels.count(self.privileged_label) for labels in orig_all_labels]
                    show_identity_num = sum(1 for P in identity_num if P > 0)
                    if fur_epoch_train == None:
                        if self.privileged_label == 0:
                            baseline_fur = baseline_fpr_train
                        else:
                            baseline_fur = baseline_fnr_train
                    else:
                        baseline_fur = fur_epoch_train

                    # loss, acc_loss, bias_loss = self.loss_function_sprt_fpr(orig_labels, predict_logits, identity_num,
                    #                                                         show_identity_num,
                    #                                                         MinValue, MaxValue, all_logits,
                    #                                                         baseline_fur, gamma)    # baseline_fpr_train

                    loss, acc_loss, bias_loss = self.loss_function_sprt_fur(orig_labels, orig_all_labels,
                                                                             predict_logits, identity_num, show_identity_num,
                                                                             MinValue, MaxValue, all_logits,
                                                                             baseline_fur, all_labels, gamma)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 打印训练信息和保存loss
                loss_dict.append(loss.item())
                print(
                    'Epoch [{}/{}], Loss: {:.4f}, acc_loss, {:.4f}, bias_loss:{:.4f}'.format(epoch + 1, self.num_epochs,
                                                                                             loss.item(),
                                                                                             acc_loss.item(),
                                                                                             bias_loss.item()))
                bias_loss_dict.append(bias_loss)
                acc_loss_dict.append(acc_loss.item())

                # new_weight = model.weight.clone()
                # new_bias = model.bias.clone()
                new_weight = model.model[0].weight.clone()
                new_bias = model.model[0].bias.clone()
                if torch.equal(new_weight, orig_weight) == True:
                    print("model weight equal.")

                classifier_new_weight = classifier.weight.clone()
                if torch.equal(classifier_new_weight, classifier_orig_weight) == True:
                    print("classifier weight equal.")

                start_index = end_index

            print("-->after epoch", epoch)

            print("-->overall baseline on training dataset")
            logtis, predict_labels = self.get_predictions_single(model, classifier, dataset_train, tokenizer, padding,
                                                                 max_seq_length,
                                                                 False)
            label = dataset_train['label'].tolist()
            TP, FP, TN, FN = self.perf_measure(label, predict_labels)
            from sklearn.metrics import accuracy_score, f1_score
            training_acc = accuracy_score(label, predict_labels)
            f1 = f1_score(label, predict_labels)
            fpr_epoch_train = FP / float(FP + TN)
            fnr_epoch_train = FN / float(FN + TP)
            print("-->ACC", training_acc)
            print("-->F1", f1)
            print("-->FPR:", fpr_epoch_train)
            print("-->FNR:", fnr_epoch_train)
            overall_acc_train.append(training_acc)
            train_acc.append(training_acc)
            train_f1.append(f1)
            train_fpr.append(fpr_epoch_train)
            train_fnr.append(fnr_epoch_train)

            print("-->overall baseline on testing dataset")
            logtis, predict_labels = self.get_predictions_single(model, classifier, dataset_test, tokenizer, padding,
                                                                 max_seq_length,
                                                                 False)
            label = dataset_test['label'].tolist()
            TP, FP, TN, FN = self.perf_measure(label, predict_labels)
            from sklearn.metrics import accuracy_score, f1_score
            testing_acc = accuracy_score(label, predict_labels)
            f1 = f1_score(label, predict_labels)
            fpr_epoch_test = FP / float(FP + TN)
            fnr_epoch_test = FN / float(FN + TP)
            print("-->ACC", testing_acc)
            print("-->F1", f1_score(label, predict_labels))
            print("-->FPR:", fpr_epoch_test)
            print("-->FNR:", fnr_epoch_test)
            overall_acc_test.append(testing_acc)
            test_acc.append(testing_acc)
            test_f1.append(f1)
            test_fpr.append(fpr_epoch_test)
            test_fnr.append(fnr_epoch_test)

            print("training dataset")
            print("-->original metric score:")
            if self.target == "individual":
                training_indi_metric, gender_metric, religion_metric, race_metric = self.get_metrics_single(model,
                                                                                                            classifier,
                                                                                                            dataset_identity,
                                                                                                            tokenizer,
                                                                                                            padding,
                                                                                                            max_seq_length,
                                                                                                            'individual')  # dataset_identity_test
                print("-->individual metric", training_indi_metric)
                ind_metrics_train.append(training_indi_metric)
                train_ind.append(training_indi_metric)
                train_gender_ind.append(gender_metric)
                train_religion_ind.append(religion_metric)
                train_race_ind.append(race_metric)

            if self.target == "group":
                training_group_metirc, train_fnr_metric, mean_fpr, mean_fnr, fpr, fnr = \
                    self.get_metrics_single(model, classifier, dataset_identity, tokenizer, padding, max_seq_length,
                                            'fnr',
                                            fpr_epoch_train, fnr_epoch_train)
                print("-->mean fnr diff metric", training_group_metirc)
                group_metrics_train.append(training_group_metirc)
                if self.target == "group":
                    train_fpr_all.append(fpr)
                    train_fnr_all.append(fnr)
                    train_fpr_mean.append(mean_fpr)
                    train_fnr_mean.append(mean_fnr)
                    train_fpr_diff.append(training_group_metirc)
                    train_fnr_diff.append(train_fnr_metric)

            print("testing dataset")
            print("-->original metric score:")
            if self.target == "individual":
                metric, gender_metric, religion_metric, race_metric = self.get_metrics_single(model, classifier,
                                                                                              dataset_identity_test,
                                                                                              tokenizer, padding,
                                                                                              max_seq_length,
                                                                                              'individual')  # dataset_identity_test
                print("-->individual metric", metric)
                ind_metrics_test.append(metric)
                test_ind.append(metric)
                test_gender_ind.append(gender_metric)
                test_religion_ind.append(religion_metric)
                test_race_ind.append(race_metric)

            if self.target == "group":
                metric, fnr_metric, mean_fpr, mean_fnr, fpr, fnr = \
                    self.get_metrics_single(model, classifier, dataset_identity_test, tokenizer, padding,
                                            max_seq_length, 'fnr',
                                            fpr_epoch_test, fnr_epoch_test)
                print("-->mean fnr diff metric", metric)
                group_metrics_test.append(metric)
                if self.target == "group":
                    test_fpr_all.append(fpr)
                    test_fnr_all.append(fnr)
                    test_fpr_mean.append(mean_fpr)
                    test_fnr_mean.append(mean_fnr)
                    test_fpr_diff.append(metric)
                    test_fnr_diff.append(fnr_metric)

            path_directory = save_dir + str(self.target) + "_" + str(self.orig_threshold) + "_" + str(gamma) + "/"
            os.makedirs(path_directory, exist_ok=True)

            model_path = path_directory + 'editor' + str(epoch) + '.pth'
            torch.save(model.state_dict(), model_path)
            model_path = path_directory + 'classifier' + str(epoch) + '.pth'
            torch.save(classifier.state_dict(), model_path)

            if self.target == "individual":
                if training_indi_metric < self.threshold and training_acc >= 0.8:
                    print("-->threshold", self.threshold)
            else:
                if training_group_metirc < self.threshold and training_acc >= 0.8:
                    print("-->threshold", self.threshold)

            exp_lr_scheduler.step()

        if self.target == "individual":
            datas = {"train_ind": train_ind, "train_gender_ind": train_gender_ind, "train_religion_ind": train_religion_ind, "train_race_ind": train_race_ind,
                     "train_acc": train_acc, "train_f1": train_f1, "train_fpr": train_fpr,  "train_fnr": train_fnr,
                     "test_ind": test_ind, "test_gender_ind": test_gender_ind, "test_religion_ind": test_religion_ind, "test_race_ind": test_race_ind,
                     "test_acc": test_acc, "test_f1": test_f1, "test_fpr": test_fpr, "test_fnr": test_fnr}
            dataset = pd.DataFrame(datas)
        else:
            datas = {"train_ind": train_ind, "train_acc": train_acc, "train_f1": train_f1, "train_fpr": train_fpr,
                     "train_fnr": train_fnr, "train_fpr_all": train_fpr_all, "train_fpr_mean": train_fpr_mean,
                     "train_fpr_diff": train_fpr_diff, "train_fnr_all": train_fnr_all, "train_fnr_mean": train_fnr_mean,
                     "train_fnr_diff": train_fnr_diff, "test_ind": test_ind, "test_acc": test_acc, "test_f1": test_f1,
                     "test_fpr": test_fpr, "test_fnr": test_fnr, "test_fpr_all": test_fpr_all, "test_fpr_mean": test_fpr_mean,
                     "test_fpr_diff": test_fpr_diff, "test_fnr_all": test_fnr_all, "test_fnr_mean": test_fnr_mean,
                     "test_fnr_diff": test_fnr_diff}
            dataset = pd.DataFrame(datas)
        print(dataset)
        return dataset
        # dataset.to_csv(path_directory + "figure/result.csv")


    def get_baseline_metrics(self, tokenizer, padding, max_seq_length):
        dataset_train = self.train_data.sample(frac=1, random_state=999)
        dataset_test = self.test_data.sample(frac=1, random_state=999)
        dataset_identity = self.train_data_identity.sample(frac=1, random_state=999)
        dataset_identity_test = self.test_data_identity.sample(frac=1, random_state=999)

        train_ind = []
        train_acc = []
        train_f1 = []
        train_fpr = []
        train_fnr = []
        train_fpr_all = []
        train_fpr_mean = []
        train_fpr_diff = []
        train_fnr_all = []
        train_fnr_mean = []
        train_fnr_diff = []

        test_ind = []
        test_acc = []
        test_f1 = []
        test_fpr = []
        test_fnr = []
        test_fpr_all = []
        test_fpr_mean = []
        test_fpr_diff = []
        test_fnr_all = []
        test_fnr_mean = []
        test_fnr_diff = []

        train_gender_ind = []
        train_religion_ind = []
        train_race_ind = []
        test_gender_ind = []
        test_religion_ind = []
        test_race_ind = []

        print("-->overall baseline on training dataset")
        logtis, predict_labels = self.get_predictions_single(None, None, dataset_train, tokenizer, padding,
                                                             max_seq_length,
                                                             False)
        label = dataset_train['label'].tolist()
        TP, FP, TN, FN = self.perf_measure(label, predict_labels)
        from sklearn.metrics import accuracy_score, f1_score
        training_acc = accuracy_score(label, predict_labels)
        f1 = f1_score(label, predict_labels)
        fpr_train = FP / float(FP + TN)
        fnr_train = FN / float(FN + TP)
        print("-->ACC", training_acc)
        print("-->F1", f1)
        print("-->FPR:", fpr_train)
        print("-->FNR:", fnr_train)
        train_acc.append(training_acc)
        train_f1.append(f1)
        train_fpr.append(fpr_train)
        train_fnr.append(fnr_train)

        print("-->overall baseline on testing dataset")
        logtis, predict_labels = self.get_predictions_single(None, None, dataset_test, tokenizer, padding,
                                                             max_seq_length,
                                                             False)
        label = dataset_test['label'].tolist()
        TP, FP, TN, FN = self.perf_measure(label, predict_labels)
        from sklearn.metrics import accuracy_score, f1_score
        testing_acc = accuracy_score(label, predict_labels)
        f1 = f1_score(label, predict_labels)
        fpr_test = FP / float(FP + TN)
        fnr_test = FN / float(FN + TP)
        print("-->ACC", testing_acc)
        print("-->F1", f1_score(label, predict_labels))
        print("-->FPR:", fpr_test)
        print("-->FNR:", fnr_test)
        test_acc.append(testing_acc)
        test_f1.append(f1)
        test_fpr.append(fpr_test)
        test_fnr.append(fnr_test)

        print("training dataset")
        print("-->original metric score:")
        training_indi_metric, gender_metric, religion_metric, race_metric = self.get_metrics_single(None, None, dataset_identity, tokenizer, padding,
                                                       max_seq_length,
                                                       'individual')  # dataset_identity_test
        print("-->individual metric", training_indi_metric)
        train_ind.append(training_indi_metric)
        train_gender_ind.append(gender_metric)
        train_religion_ind.append(religion_metric)
        train_race_ind.append(race_metric)

        training_group_metirc, train_fnr_metric, mean_fpr, mean_fnr, fpr, fnr = \
            self.get_metrics_single(None, None, dataset_identity, tokenizer, padding, max_seq_length, 'fnr',
                                    fpr_train, fnr_train)
        print("-->mean fnr diff metric", training_group_metirc)
        # group_metrics_train.append(training_group_metirc)
        train_fpr_all.append(fpr)
        train_fnr_all.append(fnr)
        train_fpr_mean.append(mean_fpr)
        train_fnr_mean.append(mean_fnr)
        train_fpr_diff.append(training_group_metirc)
        train_fnr_diff.append(train_fnr_metric)

        print("testing dataset")
        print("-->original metric score:")
        metric, gender_metric, religion_metric, race_metric = self.get_metrics_single(None, None, dataset_identity_test, tokenizer, padding, max_seq_length,
                                         'individual')  # dataset_identity_test
        print("-->individual metric", metric)
        test_ind.append(metric)
        test_gender_ind.append(gender_metric)
        test_religion_ind.append(religion_metric)
        test_race_ind.append(race_metric)

        metric, fnr_metric, mean_fpr, mean_fnr, fpr, fnr = \
            self.get_metrics_single(None, None, dataset_identity_test, tokenizer, padding, max_seq_length, 'fnr',
                                    fpr_test, fnr_test)
        print("-->mean fnr diff metric", metric)
        # group_metrics_test.append(metric)
        test_fpr_all.append(fpr)
        test_fnr_all.append(fnr)
        test_fpr_mean.append(mean_fpr)
        test_fnr_mean.append(mean_fnr)
        test_fpr_diff.append(metric)
        test_fnr_diff.append(fnr_metric)

        # datas = {"train_ind": train_ind, "train_acc": train_acc, "train_f1": train_f1, "train_fpr": train_fpr,
        #          "train_fnr": train_fnr, "train_fpr_all": train_fpr_all, "train_fpr_mean": train_fpr_mean,
        #          "train_fpr_diff": train_fpr_diff, "train_fnr_all": train_fnr_all, "train_fnr_mean": train_fnr_mean,
        #          "train_fnr_diff": train_fnr_diff, "test_ind": test_ind, "test_acc": test_acc, "test_f1": test_f1,
        #          "test_fpr": test_fpr, "test_fnr": test_fnr, "test_fpr_all": test_fpr_all,
        #          "test_fpr_mean": test_fpr_mean,
        #          "test_fpr_diff": test_fpr_diff, "test_fnr_all": test_fnr_all, "test_fnr_mean": test_fnr_mean,
        #          "test_fnr_diff": test_fnr_diff}
        # datas = {"train_gender_ind": train_gender_ind, "train_religion_ind": train_religion_ind, "train_race_ind": train_race_ind,
        #          "test_gender_ind": test_gender_ind, "test_religion_ind": test_religion_ind, "test_race_ind": test_race_ind}
        datas = {"train_ind": train_ind, "train_gender_ind": train_gender_ind, "train_religion_ind": train_religion_ind,
                 "train_race_ind": train_race_ind, "train_acc": train_acc, "train_f1": train_f1, "train_fpr": train_fpr,
                 "train_fnr": train_fnr, "test_ind": test_ind, "test_gender_ind": test_gender_ind,
                 "test_religion_ind": test_religion_ind, "test_race_ind": test_race_ind, "test_acc": test_acc,
                 "test_f1": test_f1, "test_fpr": test_fpr, "test_fnr": test_fnr}
        dataset = pd.DataFrame(datas)

        return dataset

    def get_baseline_metrics_with_pred(self, dataset, predict_labels):

        train_ind = []
        train_acc = []
        train_f1 = []
        train_fpr = []
        train_fnr = []
        train_fpr_all = []
        train_fpr_mean = []
        train_fpr_diff = []
        train_fnr_all = []
        train_fnr_mean = []
        train_fnr_diff = []

        test_ind = []
        test_acc = []
        test_f1 = []
        test_fpr = []
        test_fnr = []
        test_fpr_all = []
        test_fpr_mean = []
        test_fpr_diff = []
        test_fnr_all = []
        test_fnr_mean = []
        test_fnr_diff = []

        train_gender_ind = []
        train_religion_ind = []
        train_race_ind = []
        test_gender_ind = []
        test_religion_ind = []
        test_race_ind = []


        print("-->overall baseline")
        # logtis, predict_labels = self.get_predictions_single(None, None, dataset_test, tokenizer, padding,
        #                                                      max_seq_length,
        #                                                      False)
        label = dataset['label'].tolist()
        TP, FP, TN, FN = self.perf_measure(label, predict_labels)
        from sklearn.metrics import accuracy_score, f1_score
        testing_acc = accuracy_score(label, predict_labels)
        f1 = f1_score(label, predict_labels)
        fpr_test = FP / float(FP + TN)
        fnr_test = FN / float(FN + TP)
        print("-->ACC", testing_acc)
        print("-->F1", f1_score(label, predict_labels))
        print("-->FPR:", fpr_test)
        print("-->FNR:", fnr_test)
        test_acc.append(testing_acc)
        test_f1.append(f1)
        test_fpr.append(fpr_test)
        test_fnr.append(fnr_test)

        # print("testing dataset")
        # print("-->original metric score:")
        # metric, gender_metric, religion_metric, race_metric = self.get_metrics_single(None, None, dataset, tokenizer, padding, max_seq_length,
        #                                  'individual')  # dataset_identity_test
        # print("-->individual metric", metric)
        # test_ind.append(metric)
        # test_gender_ind.append(gender_metric)
        # test_religion_ind.append(religion_metric)
        # test_race_ind.append(race_metric)

        metric, fnr_metric, mean_fpr, mean_fnr, fpr, fnr = \
            self.get_metrics_single_with_pred(dataset, predict_labels, 'fnr', fpr_test, fnr_test)
        print("-->mean fnr diff metric", metric)
        # group_metrics_test.append(metric)
        test_fpr_all.append(fpr)
        test_fnr_all.append(fnr)
        test_fpr_mean.append(mean_fpr)
        test_fnr_mean.append(mean_fnr)
        test_fpr_diff.append(metric)
        test_fnr_diff.append(fnr_metric)

        datas = {"test_acc": test_acc, "test_f1": test_f1,
                 "test_fpr": test_fpr, "test_fnr": test_fnr, "test_fpr_all": test_fpr_all,
                 "test_fpr_mean": test_fpr_mean,
                 "test_fpr_diff": test_fpr_diff, "test_fnr_all": test_fnr_all, "test_fnr_mean": test_fnr_mean,
                 "test_fnr_diff": test_fnr_diff}
        # datas = {"train_gender_ind": train_gender_ind, "train_religion_ind": train_religion_ind, "train_race_ind": train_race_ind,
        #          "test_gender_ind": test_gender_ind, "test_religion_ind": test_religion_ind, "test_race_ind": test_race_ind}
        # datas = {"train_ind": train_ind, "train_gender_ind": train_gender_ind, "train_religion_ind": train_religion_ind,
        #          "train_race_ind": train_race_ind, "train_acc": train_acc, "train_f1": train_f1, "train_fpr": train_fpr,
        #          "train_fnr": train_fnr, "test_ind": test_ind, "test_gender_ind": test_gender_ind,
        #          "test_religion_ind": test_religion_ind, "test_race_ind": test_race_ind, "test_acc": test_acc,
        #          "test_f1": test_f1, "test_fpr": test_fpr, "test_fnr": test_fnr}
        dataset = pd.DataFrame(datas)

        return dataset

    def get_all_metrics(self, model, classifier, path_directory, tokenizer, padding, max_seq_length):
        dataset_train = self.train_data.sample(frac=1, random_state=999)
        dataset_test = self.test_data.sample(frac=1, random_state=999)
        dataset_identity = self.train_data_identity.sample(frac=1, random_state=999)
        dataset_identity_test = self.test_data_identity.sample(frac=1, random_state=999)

        train_ind = []
        train_gender_ind = []
        train_religion_ind = []
        train_race_ind = []
        test_ind = []
        test_gender_ind = []
        test_religion_ind = []
        test_race_ind = []
        train_acc = []
        train_f1 = []
        train_FPR = []
        train_FNR = []
        test_acc = []
        test_f1 = []
        test_FPR = []
        test_FNR = []
        gender_accs = []
        religion_accs = []
        race_accs = []

        # print("training dataset")
        # print("-->original metric score:")
        # metric = self.get_metrics_single(None, None, dataset_identity, tokenizer, padding, max_seq_length,
        #                                  'individual')  # dataset_identity_test
        # print("-->individual metric", metric)
        # train_ind.append(metric)
        #
        # print("testing dataset")
        # print("-->original metric score:")
        # metric = self.get_metrics_single(None, None, dataset_identity_test, tokenizer, padding, max_seq_length,
        #                                  'individual')  # dataset_identity_test
        # print("-->individual metric", metric)
        # test_ind.append(metric)
        #
        #
        # print("-->overall baseline on training dataset")
        # logtis, predict_labels = self.get_predictions_single(None, None, dataset_train, tokenizer, padding,
        #                                                      max_seq_length,
        #                                                      False)
        # label = dataset_train['label'].tolist()
        # TP, FP, TN, FN = self.perf_measure(label, predict_labels)
        # acc = accuracy_score(label, predict_labels)
        # train_acc.append(acc)
        # print("-->ACC", acc)
        # f1 = f1_score(label, predict_labels)
        # train_f1.append(f1)
        # print("-->F1", f1)
        # FPR = FP / float(FP + TN)
        # train_FPR.append(FPR)
        # print("-->FPR:", FPR)
        # FNR = FN / float(FN + TP)
        # train_FNR.append(FNR)
        # print("-->FNR:", FNR)
        #
        # print("-->overall baseline on testing dataset")
        # logtis, predict_labels = self.get_predictions_single(None, None, dataset_test, tokenizer, padding,
        #                                                      max_seq_length,
        #                                                      False)
        # label = dataset_test['label'].tolist()
        # TP, FP, TN, FN = self.perf_measure(label, predict_labels)
        # acc = accuracy_score(label, predict_labels)
        # test_acc.append(acc)
        # print("-->ACC", acc)
        # f1 = f1_score(label, predict_labels)
        # test_f1.append(f1)
        # print("-->F1", f1)
        # FPR = FP / float(FP + TN)
        # test_FPR.append(FPR)
        # print("-->FPR:", FPR)
        # FNR = FN / float(FN + TP)
        # test_FNR.append(FNR)
        # print("-->FNR:", FNR)

        # epochs = list(range(0, 20))
        epochs = [10]
        for epoch in epochs:
            print("-->epoch", epoch)
            model_path = path_directory + 'editor' + str(epoch) + '.pth'
            print("-->model_path", model_path)
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)

            model_path = path_directory + 'classifier' + str(epoch) + '.pth'
            state_dict = torch.load(model_path)
            classifier.load_state_dict(state_dict)

            # print("-->overall baseline on training dataset")
            # logtis, predict_labels = self.get_predictions_single(model, classifier, dataset_train, tokenizer, padding,
            #                                                      max_seq_length,
            #                                                      False)
            # label = dataset_train['label'].tolist()
            # TP, FP, TN, FN = self.perf_measure(label, predict_labels)
            # acc = accuracy_score(label, predict_labels)
            # train_acc.append(acc)
            # print("-->ACC", acc)
            # f1 = f1_score(label, predict_labels)
            # train_f1.append(f1)
            # print("-->F1", f1)
            # FPR = FP / float(FP + TN)
            # train_FPR.append(FPR)
            # print("-->FPR:", FPR)
            # FNR = FN / float(FN + TP)
            # train_FNR.append(FNR)
            # print("-->FNR:", FNR)
            #
            print("-->overall baseline on testing dataset")
            logtis, predict_labels = self.get_predictions_single(model, classifier, dataset_test, tokenizer, padding,
                                                                 max_seq_length,
                                                                 False)
            label = dataset_test['label'].tolist()
            TP, FP, TN, FN = self.perf_measure(label, predict_labels)
            acc = accuracy_score(label, predict_labels)
            test_acc.append(acc)
            print("-->ACC", acc)
            f1 = f1_score(label, predict_labels)
            test_f1.append(f1)
            print("-->F1", f1)
            FPR = FP / float(FP + TN)
            test_FPR.append(FPR)
            print("-->FPR:", FPR)
            FNR = FN / float(FN + TP)
            test_FNR.append(FNR)
            print("-->FNR:", FNR)

            # print("training dataset")
            # metric, gender_metric, religion_metric, race_metric = self.get_metrics_single(model, classifier, dataset_identity, tokenizer, padding, max_seq_length,
            #                                  'individual')  # dataset_identity_test
            # print("-->individual metric", metric)
            # train_ind.append(metric)
            # train_gender_ind.append(gender_metric)
            # train_religion_ind.append(religion_metric)
            # train_race_ind.append(race_metric)

            test_fpr_metric, test_fnr_metric, test_mean_fpr, test_mean_fnr, test_fpr, test_fnr, \
            gender_acc, religion_acc, race_acc, gender_fpr, religion_fpr, race_fpr,\
            gender_fnr, religion_fnr, race_fnr = self.get_metrics_single(model, classifier, dataset_identity_test,
                                                                         tokenizer, padding, max_seq_length, 'fnr',
                                                                         FPR, FNR, True)
            print("-->mean fnr diff metric", test_fpr_metric)
            # test_fpr_mean.append(test_mean_fpr)
            # test_fnr_mean.append(test_mean_fnr)
            # test_fpr_diff.append(test_fpr_metric)
            # test_fnr_diff.append(test_fnr_metric)
            gender_accs.append(gender_acc)
            religion_accs.append(religion_acc)
            race_accs.append(race_acc)

            # print("testing dataset")
            # metric, gender_metric, religion_metric, race_metric = self.get_metrics_single(model, classifier, dataset_identity_test, tokenizer, padding, max_seq_length,
            #                                  'individual')  # dataset_identity_test
            # print("-->individual metric", metric)
            # test_ind.append(metric)
            # test_gender_ind.append(gender_metric)
            # test_religion_ind.append(religion_metric)
            # test_race_ind.append(race_metric)

        # datas = {"train_ind": train_ind, "train_gender_ind": train_gender_ind, "train_religion_ind": train_religion_ind,
        #          "train_race_ind": train_race_ind, "train_acc": train_acc, "train_f1": train_f1, "train_FPR": train_FPR,
        #          "train_FNR": train_FNR, "test_ind": test_ind, "test_gender_ind": test_gender_ind, "test_religion_ind": test_religion_ind,
        #          "test_race_ind": test_race_ind,
        #          "test_acc": test_acc, "test_f1": test_f1, "test_FPR": test_FPR, "test_FNR": test_FNR}
        # datas = {"train_gender_ind": train_gender_ind, "train_religion_ind": train_religion_ind,
                 # "train_race_ind": train_race_ind, "test_gender_ind": test_gender_ind,
                 # "test_religion_ind": test_religion_ind, "test_race_ind": test_race_ind}
        datas = {'test_acc': test_acc, 'test_f1': test_f1, 'test_fpr': test_FPR, 'test_fnr': test_FNR,
                 'gender_acc': gender_accs, 'religion_acc': religion_accs, 'race_acc': race_accs}
        dataset = pd.DataFrame(datas)
        return dataset



    def get_texts_with_identity(self, dataset):
        texts = dataset['text'].values.tolist()
        labels = dataset['label'].values.tolist()
        texts_with_identity = []
        relate_labels = []
        all_exists_terms = []
        for j in list(range(0, len(texts))): #tqdm
            text = texts[j]
            label = labels[j]
            all_i = []
            terms = []
            for i in range(0, len(identities)):
                identity = identities[i]
                if_exist, exist_terms = identity_detect_with_term(text, identity)
                if if_exist == True:
                    terms = terms + exist_terms
                    all_i.append(i)
                if i == len(identities) - 1:
                    if all_i == []:
                        pass
                    else:
                        texts_with_identity.append(text)
                        relate_labels.append(label)
                        all_exists_terms.append(terms)
        return texts_with_identity, all_exists_terms, relate_labels

    def process_dataset(self, texts, all_exists_terms, identity_class):
        new_texts = []
        terms = term_class_dict[identity_class]
        for i in range(len(texts)):
            text = texts[i]
            exists_terms = all_exists_terms[i]
            for t in exists_terms:
                text = text.replace(t, random.choice(terms))
            new_texts.append(text)
        return new_texts

    def get_texts_with_given_identity(self, dataset, identity_class):
        texts_with_identity, all_exists_terms, relate_labels = self.get_texts_with_identity(dataset)
        new_texts = self.process_dataset(texts_with_identity, all_exists_terms, identity_class)
        return new_texts, relate_labels

    def get_all_texts_with_identity(self, dataset):
        def Merge(dict1, dict2):
            res = {**dict1, **dict2}
            return res
        dataset_identity = {}
        for i in range(0, len(identities)):
            identity = identities[i]
            new_texts, relate_labels = self.get_texts_with_given_identity(dataset, identity)
            dict = {identity: {"text": new_texts, "label": relate_labels}}
            dataset_identity = Merge(dataset_identity, dict)
        return dataset_identity

    def perf_measure(self, y_actual, y_pred):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_pred)):
            if y_actual[i] == y_pred[i] == 1:
                TP += 1
            if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
                FP += 1
            if y_actual[i] == y_pred[i] == 0:
                TN += 1
            if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
                FN += 1

        return TP, FP, TN, FN

    def get_metrics(self, model, dataset_identity, tokenizer, padding, max_seq_length, metric):
        # model = self.CustomModel.custom_model["custom_module"]
        all_FPR = []
        all_FNR = []
        all_metrics = {'acc':[], 'fpr':[], 'fnr':[]}
        for i in range(len(identities)):
            identity = identities[i]
            print("-->identity", identity)

            datas = {'text':dataset_identity[identity], 'label':dataset_identity['label']}
            dataset = DataFrame(datas)

            logits, labels = self.get_predictions(model=model, dataset=dataset, tokenizer=tokenizer, padding=padding,
                                                                     max_seq_length=max_seq_length, if_identity=False)


            predict_labels = labels
            base_labels = dataset['label']
            # base_labels = datas['label']
            select_base_labels = np.array(base_labels)
            select_predict_labels = np.array(predict_labels)

            if len(dataset['text']) == 0:
                break

            TP, FP, TN, FN = self.perf_measure(select_base_labels, select_predict_labels)

            FPR = FP / (FP + TN)
            FNR = FN / (TP + FN)
            # FDR = FP / (TP + FP)
            ACC = (TP + TN) / (TP + FP + FN + TN)
            print("-->ACC", ACC)
            all_metrics['acc'].append(ACC)
            all_metrics['fpr'].append(FPR)
            all_metrics['fnr'].append(FNR)
            # all_FPR.append(FPR)
            # all_FNR.append(FNR)
            print("-->FPR", FPR)
            print("-->FNR", FNR)
            print("-->positive probability:{}, {}/{}".format(list(predict_labels).count(1) / len(predict_labels),
                                                             list(predict_labels).count(1), len(predict_labels)))
        print("-->FPR:{}, mean FPR:{}".format(all_metrics['fpr'], np.mean(np.array(all_metrics['fpr']))))
        print("-->FNR:{}, mean FNR:{}".format( all_metrics['fnr'], np.mean(np.array(all_metrics['fnr']))))
        return np.mean(np.array(all_metrics[metric]))

    def get_metrics_individual(self, model, dataset_identity, tokenizer, padding, max_seq_length):
        all_logits = []
        # all_logits = torch.FloatTensor([])
        for i in range(len(identities)):
            identity = identities[i]
            print("-->identity", identity)

            datas = {'text': dataset_identity[identity], 'label': dataset_identity['label']}
            dataset = DataFrame(datas)

            logits, labels = self.get_predictions(model=model, dataset=dataset, tokenizer=tokenizer,
                                                         padding=padding,
                                                         max_seq_length=max_seq_length, if_identity=False)
            all_logits.append(logits)
            # all_logits += logits

        sum_logits = all_logits[0]
        for i in range(1, len(identities)):
            sum_logits = torch.add(sum_logits, all_logits[i])
        mean_logits = torch.mean(sum_logits)
        print("-->mean_logits", mean_logits)
        print("-->last logits", logits)
        return logits
        # return mean_logits

    def get_metrics_single_category(self, model, classifier, dataset_identity, tokenizer, padding, max_seq_length, metric,
                                    baseline_fpr_train=None, baseline_fnr_train=None):
        if metric == 'individual':
            ID = IdentityDetect()
            num_idi = 0
            gender_num = 0
            gender_idi = 0
            religion_num = 0
            religion_idi = 0
            race_num = 0
            race_idi = 0
            for i in range(len(dataset_identity['orig_text'])):
                orig_text = dataset_identity['orig_text'][i]
                idis = eval(dataset_identity['idis'][i])
                label = dataset_identity['label'][i]
                exist_idi = False

                # detect identity category
                if ID.identity_category_detect(orig_text, gender) == True:
                    gender_num += 1
                elif ID.identity_category_detect(orig_text, religion) == True:
                    religion_num += 1
                elif ID.identity_category_detect(orig_text, race) == True:
                    race_num += 1

                datas = {'text': [orig_text], 'label': [label]}
                dataset = DataFrame(datas)
                logits, predict_label = self.get_predictions_single(model=model, classifier=classifier, dataset=dataset,
                                                                    tokenizer=tokenizer,
                                                                    padding=padding,
                                                                    max_seq_length=max_seq_length, if_identity=False)
                orig_predict_label = predict_label[0]
                for idi in idis:
                    datas = {'text': [idi], 'label': [label]}
                    dataset = DataFrame(datas)
                    logits, predict_label = self.get_predictions_single(model=model, classifier=classifier, dataset=dataset, tokenizer=tokenizer,
                                                                 padding=padding,
                                                                 max_seq_length=max_seq_length, if_identity=False)
                    if predict_label[0] != orig_predict_label:
                        if ID.identity_category_detect(idi, gender) == True:
                            gender_idi += 1
                        elif ID.identity_category_detect(idi, religion) == True:
                            religion_idi += 1
                        elif ID.identity_category_detect(idi, race) == True:
                            race_idi += 1
                        exist_idi = True
                        # num_idi += 1
                        break

                if exist_idi == True:
                    num_idi += 1
            individual_metric = num_idi / float(len(dataset_identity['orig_text']))
            gender_metric = gender_idi / float(gender_num)
            religion_metric = religion_idi / float(religion_num)
            race_metric = race_idi / float(race_num)
            print("-->individual metric:{}, num_idi:{}, num_all:{}".format(individual_metric, num_idi, len(dataset_identity['orig_text'])))
            print("-->individual metric gender:{}, num_idi:{}, num_all:{}".format(gender_metric, gender_idi, gender_num))
            print("-->individual metric religion:{}, num_idi:{}, num_all:{}".format(religion_metric, religion_idi, religion_num))
            print("-->individual metric race:{}, num_idi:{}, num_all:{}".format(race_metric, race_idi, race_num))
            return individual_metric, gender_metric, religion_metric, race_metric

        elif metric == 'fnr' or metric == 'fpr':
            all_metrics = {'acc': [], 'fpr': [], 'fnr': []}
            # ----------group metric on orig_text----------
            datas = {'text': dataset_identity['orig_text'], 'label': dataset_identity['label']}
            dataset = DataFrame(datas)
            texts = dataset['text'].tolist()
            base_labels = dataset['label'].tolist()

            logits, predict_labels = self.get_predictions_single(model=model, classifier=classifier, dataset=dataset, tokenizer=tokenizer,
                                                         padding=padding, max_seq_length=max_seq_length, if_identity=False)
            print("-->all_length:", len(predict_labels))
            labels_orig = []
            labels_predict = []
            labels_truth = []
            ID = IdentityDetect()
            for j in range(0, len(ID.identities)):
                identity = ID.identities[j]
                select_labels_truth = []
                select_labels_predict = []
                for i in range(0, len(texts)):
                    # text = dataset['text'][i]
                    text = texts[i]
                    if ID.identity_detect(text, identity) == True:
                        # label = dataset['label'][i]
                        label = base_labels[i]
                        predict_label = predict_labels[i]
                        select_labels_truth.append(label)
                        select_labels_predict.append(predict_label)
                labels_truth.append(select_labels_truth)
                labels_predict.append(select_labels_predict)

            for i in range(0, len(ID.identities)):
                identity = ID.identities[i]
                print("-->identity", identity)
                print("-->length", len(labels_truth[i]))
                TP, FP, TN, FN = self.perf_measure(labels_truth[i], labels_predict[i])
                try:
                    FPR = FP / (FP + TN)
                except:
                    print("(FP + TN) is 0")
                    FPR = None
                try:
                      FNR = FN / (TP + FN)
                except:
                    print("(TP + FN) is 0")
                    FNR = None
                ACC = accuracy_score(labels_truth[i], labels_predict[i])
                print("-->ACC", ACC)
                print("-->F1", f1_score(labels_truth[i], labels_predict[i]))
                all_metrics['acc'].append(ACC)
                all_metrics['fpr'].append(FPR)
                all_metrics['fnr'].append(FNR)
                print("-->FPR", FPR)
                print("-->FNR", FNR)
                print("-->positive probability:{}, {}/{}".format(list(labels_predict[i]).count(1) / len(labels_predict[i]),
                                                                 list(labels_predict[i]).count(1), len(labels_predict[i])))
                print("-->group-truth label positive probability:{}, {}/{}".format(list(labels_truth[i]).count(1) / len(labels_truth[i]),
                                                                 list(labels_truth[i]).count(1), len(labels_truth[i])))

            all_metrics['fpr'] = [x for x in all_metrics['fpr'] if x != None]
            all_metrics['fnr'] = [x for x in all_metrics['fnr'] if x != None]

            fpr_diff = [abs(fpr - baseline_fpr_train) for fpr in all_metrics['fpr']]
            print("-->fpr_diff", fpr_diff)
            fnr_diff = [abs(fnr - baseline_fnr_train) for fnr in all_metrics['fnr']]
            print("-->fnr_diff", fnr_diff)

            print("-->FPR:{}, mean FPR:{}, mean FPR diff:{}".format(all_metrics['fpr'], np.mean(np.array(all_metrics['fpr'])), np.mean(np.array(fpr_diff))))
            print("-->FNR:{}, mean FNR:{}, mean FNR diff:{}".format(all_metrics['fnr'], np.mean(np.array(all_metrics['fnr'])), np.mean(np.array(fnr_diff))))
            return np.mean(np.array(fpr_diff)), np.mean(np.array(fnr_diff)), np.mean(np.array(all_metrics['fpr'])),\
               np.mean(np.array(all_metrics['fnr'])), all_metrics['fpr'], all_metrics['fnr']

    def get_base_performance(self, labels_truth, labels_predict):
        TP, FP, TN, FN = self.perf_measure(labels_truth, labels_predict)
        try:
            FPR = FP / (FP + TN)
        except:
            print("(FP + TN) is 0")
            FPR = None
        try:
            FNR = FN / (TP + FN)
        except:
            print("(TP + FN) is 0")
            FNR = None
        ACC = accuracy_score(labels_truth, labels_predict)
        F1 = f1_score(labels_truth, labels_predict)
        return ACC, F1, FPR, FNR

    def get_metrics_single(self, model, classifier, dataset_identity, tokenizer, padding, max_seq_length, metric,
                           baseline_fpr_train=None, baseline_fnr_train=None, if_get_category_acc=False):
        # model = self.CustomModel.custom_model["custom_module"]
        all_FPR = []
        all_FNR = []
        if metric == 'individual pro diff':
            all_logits = []
            all_labels = []
            for i in range(len(identities)):
                identity = identities[i]
                print("-->identity", identity)

                datas = {'text': dataset_identity[identity], 'label': dataset_identity['label']}
                dataset = DataFrame(datas)

                logits, labels = self.get_predictions_single(model=model, classifier=classifier, dataset=dataset,
                                                             tokenizer=tokenizer, padding=padding,
                                                             max_seq_length=max_seq_length, if_identity=False)
                all_logits.append(logits)
            texts = dataset_identity['text'].values.tolist()
            all_logits = np.array(all_logits)
            all_mean_logits = []
            for j in range(0, len(texts)):
                logits = all_logits[:,j]
                mean_logtis = np.mean(logits)
                mean_diff = np.mean([np.abs(l-mean_logtis) for l in logits])
                all_mean_logits.append(mean_diff)
            individual_metric = np.mean(np.array(all_mean_logits))
            print("-->individual metric:", individual_metric)
            return individual_metric
        elif metric == 'individual':
            ID = IdentityDetect()
            num_idi = 0
            diff_i = []
            orig_identities = []
            for i in range(len(dataset_identity['orig_text'])):
                one_orig_text_idi = []
                orig_text = dataset_identity['orig_text'][i]
                idis = eval(dataset_identity['idis'][i])
                label = dataset_identity['label'][i]

                datas = {'text': [orig_text], 'label': [label]}
                dataset = DataFrame(datas)
                logits, predict_label = self.get_predictions_single(model=model, classifier=classifier, dataset=dataset,
                                                                    tokenizer=tokenizer,
                                                                    padding=padding,
                                                                    max_seq_length=max_seq_length, if_identity=False)
                orig_predict_label = predict_label[0]
                get_idi = False
                contain_i_orig = ID.which_identity(orig_text)
                orig_identities.append(contain_i_orig)
                for idi in idis:
                    datas = {'text': [idi], 'label': [label]}
                    dataset = DataFrame(datas)
                    logits, predict_label = self.get_predictions_single(model=model, classifier=classifier, dataset=dataset, tokenizer=tokenizer,
                                                                 padding=padding,
                                                                 max_seq_length=max_seq_length, if_identity=False)
                    if predict_label[0] != orig_predict_label:
                        if get_idi == False:
                            num_idi += 1
                        # contain_i_orig = ID.which_identity(orig_text)
                        contain_i_idi = ID.which_identity(idi)

                        different_identity = []
                        for identity in contain_i_orig:
                            if identity not in contain_i_idi:
                                different_identity.append(identity)
                        for identity in contain_i_idi:
                            if identity not in contain_i_orig:
                                different_identity.append(identity)

                        one_orig_text_idi.append(different_identity)
                        get_idi = True
                        # break
                diff_i.append(one_orig_text_idi)
                #     all_idi_labels.append(predict_label[0])
                # if sum(all_idi_labels)/float(len(idis)+1) != 1 and sum(all_idi_labels)/float(len(idis)+1) != 0:
                #     num_idi += 1
            individual_metric = num_idi / float(len(dataset_identity['orig_text']))
            print("-->individual metric:{}, num_idi:{}, num_all:{}".format(individual_metric, num_idi, len(dataset_identity['orig_text'])))

            gender_num = 0
            religion_num = 0
            race_num = 0
            for contain_identity in orig_identities:
                for identity in contain_identity:
                    if identity in ID.identity_category['gender']:
                        gender_num += 1
                    elif identity in ID.identity_category['religion']:
                        religion_num += 1
                    elif identity in ID.identity_category['race']:
                        race_num += 1

            gender_num_idi = 0
            religion_num_idi = 0
            race_num_idi = 0
            for one_samples_i in diff_i:
                gender_exist = False
                religion_exist = False
                race_exist = False
                for diff_identity in one_samples_i:  # e.g., diff_identity: ['male', 'homosexual']
                    if all(i in ID.identity_category['gender'] for i in diff_identity) and gender_exist == False:
                        gender_num_idi += 1
                        gender_exist = True
                    elif all(i in ID.identity_category['religion'] for i in diff_identity) and religion_exist == False:
                        religion_num_idi += 1
                        religion_exist = True
                    elif all(i in ID.identity_category['race'] for i in diff_identity) and race_exist == False:
                        race_num_idi += 1
                        race_exist = True

            gender_metric = gender_num_idi/float(gender_num)
            religion_metric = religion_num_idi/float(religion_num)
            race_metric = race_num_idi/float(race_num)
            print("-->individual metric for gender:{}, num_idi:{}, num_all:{}".format(gender_metric, gender_num_idi,
                                                                                      gender_num))
            print("-->individual metric for religion:{}, num_idi:{}, num_all:{}".format(religion_metric, religion_num_idi,
                                                                           religion_num))
            print("-->individual metric for race:{}, num_idi:{}, num_all:{}".format(race_metric, race_num_idi,
                                                                           race_num))
            # print("-->diff_i", diff_i)
            # with open("diff_identity_train.txt", "w") as f:
            #     for one_idi_identity in diff_i:
            #         # Write each item to the file, followed by a newline character
            #         f.write(f"{str(one_idi_identity)}\n")
            return individual_metric, gender_metric, religion_metric, race_metric
        elif metric == 'fnr' or metric == 'fpr':
            all_metrics = {'acc': [], 'fpr': [], 'fnr': []}
            # ----------group metric on orig_text----------
            datas = {'text': dataset_identity['orig_text'], 'label': dataset_identity['label']}
            dataset = DataFrame(datas)
            texts = dataset['text'].tolist()
            base_labels = dataset['label'].tolist()

            # # ----------group metric on all_texts----------
            # all_texts = []
            # all_labels = []
            # for i in range(len(dataset_identity['orig_text'])):
            #     orig_text = dataset_identity['orig_text'][i]
            #     idis = eval(dataset_identity['idis'][i])
            #     label = dataset_identity['label'][i]
            #     all_texts.append(orig_text)
            #     all_labels.append(label)
            #     for idi in idis:
            #         all_texts.append(idi)
            #         all_labels.append(label)
            # datas = {'text': all_texts, 'label': all_labels}
            # dataset = DataFrame(datas)
            # print("-->dataset", dataset)

            logits, predict_labels = self.get_predictions_single(model=model, classifier=classifier, dataset=dataset, tokenizer=tokenizer,
                                                         padding=padding, max_seq_length=max_seq_length, if_identity=False)
            print("-->all_length:", len(predict_labels))

            labels_orig = []
            labels_predict = []
            labels_truth = []
            ID = IdentityDetect()
            for j in range(0, len(ID.identities)):
                identity = ID.identities[j]
                select_labels_truth = []
                select_labels_predict = []
                for i in range(0, len(texts)):
                    # text = dataset['text'][i]
                    text = texts[i]
                    if ID.identity_detect(text, identity) == True:
                        # label = dataset['label'][i]
                        label = base_labels[i]
                        predict_label = predict_labels[i]
                        select_labels_truth.append(label)
                        select_labels_predict.append(predict_label)
                labels_truth.append(select_labels_truth)
                labels_predict.append(select_labels_predict)

            if if_get_category_acc == True:
                gender_labels_truth = labels_truth[0] + labels_truth[1] + labels_truth[2]
                gender_labels_predict = labels_predict[0] + labels_predict[1] + labels_predict[2]
                religion_labels_truth = labels_truth[3] + labels_truth[4] + labels_truth[5]
                religion_labels_predict = labels_predict[3] + labels_predict[4] + labels_predict[5]
                race_labels_truth = labels_truth[6] + labels_truth[7]
                race_labels_predict = labels_predict[6] + labels_predict[7]
                gender_acc, gender_f1, gender_fpr, gender_fnr = self.get_base_performance(gender_labels_truth,
                                                                                          gender_labels_predict)
                religion_acc, religion_f1, religion_fpr, religion_fnr = self.get_base_performance(religion_labels_truth,
                                                                                                  religion_labels_predict)
                race_acc, race_f1, race_fpr, race_fnr = self.get_base_performance(race_labels_truth,
                                                                                  race_labels_predict)

            for i in range(0, len(ID.identities)):
                identity = ID.identities[i]
                print("-->identity", identity)
                print("-->length", len(labels_truth[i]))
                TP, FP, TN, FN = self.perf_measure(labels_truth[i], labels_predict[i])
                try:
                    FPR = FP / (FP + TN)
                except:
                    print("(FP + TN) is 0")
                    FPR = None
                try:
                      FNR = FN / (TP + FN)
                except:
                    print("(TP + FN) is 0")
                    FNR = None
                ACC = accuracy_score(labels_truth[i], labels_predict[i])
                print("-->ACC", ACC)
                print("-->F1", f1_score(labels_truth[i], labels_predict[i]))
                all_metrics['acc'].append(ACC)
                all_metrics['fpr'].append(FPR)
                all_metrics['fnr'].append(FNR)
                print("-->FPR", FPR)
                print("-->FNR", FNR)
                print("-->positive probability:{}, {}/{}".format(list(labels_predict[i]).count(1) / len(labels_predict[i]),
                                                                 list(labels_predict[i]).count(1), len(labels_predict[i])))
                print("-->group-truth label positive probability:{}, {}/{}".format(list(labels_truth[i]).count(1) / len(labels_truth[i]),
                                                                 list(labels_truth[i]).count(1), len(labels_truth[i])))

            all_metrics['fpr'] = [x for x in all_metrics['fpr'] if x != None]
            all_metrics['fnr'] = [x for x in all_metrics['fnr'] if x != None]

            fpr_diff = [abs(fpr - baseline_fpr_train) for fpr in all_metrics['fpr']]
            print("-->fpr_diff", fpr_diff)
            fnr_diff = [abs(fnr - baseline_fnr_train) for fnr in all_metrics['fnr']]
            print("-->fnr_diff", fnr_diff)

            print("-->FPR:{}, mean FPR:{}, mean FPR diff:{}".format(all_metrics['fpr'], np.mean(np.array(all_metrics['fpr'])), np.mean(np.array(fpr_diff))))
            print("-->FNR:{}, mean FNR:{}, mean FNR diff:{}".format(all_metrics['fnr'], np.mean(np.array(all_metrics['fnr'])), np.mean(np.array(fnr_diff))))
            if if_get_category_acc == True:
                return np.mean(np.array(fpr_diff)), np.mean(np.array(fnr_diff)), \
                   np.mean(np.array(all_metrics['fpr'])), np.mean(np.array(all_metrics['fnr'])), \
                   all_metrics['fpr'], all_metrics['fnr'], gender_acc, religion_acc, race_acc, \
                       gender_fpr, religion_fpr, race_fpr, gender_fnr, religion_fnr, race_fnr
            else:
                return np.mean(np.array(fpr_diff)), np.mean(np.array(fnr_diff)), \
                       np.mean(np.array(all_metrics['fpr'])), np.mean(np.array(all_metrics['fnr'])), \
                       all_metrics['fpr'], all_metrics['fnr']
        else:
            # ----------group metric on orig_text----------
            datas = {'text': dataset_identity['orig_text'], 'label': dataset_identity['label']}
            dataset = DataFrame(datas)

            labels_orig = []
            ID = IdentityDetect()
            for j in range(0, len(ID.identities)):
                identity = ID.identities[j]
                print("-->identity", identity)
                select_labels_orig = []
                for i in range(0, len(dataset_identity['orig_text'])):
                    text = dataset_identity['orig_text'][i]
                    if ID.identity_detect(text, identity) == True:
                        label = dataset_identity['label'][i]
                        select_labels_orig.append(label)
                labels_orig.append(select_labels_orig)

            for i in range(0, len(ID.identities)):
                labels_orig_one_identity = labels_orig[i]
                identity = ID.identities[i]
                print("-->identity", identity)
                print("-->length", len(labels_orig[i]))
                print("-->positive probability:{}, {}/{}".format(list(labels_orig_one_identity).count(1) / len(labels_orig_one_identity),
                                                                 list(labels_orig_one_identity).count(1), len(labels_orig_one_identity)))

            return

    def get_metrics_single_with_pred(self, dataset_identity, predict_labels, metric, baseline_fpr_train=None, baseline_fnr_train=None):

        if metric == 'individual':
            ID = IdentityDetect()
            num_idi = 0
            diff_i = []
            orig_identities = []
            for i in range(len(dataset_identity['orig_text'])):
                one_orig_text_idi = []
                orig_text = dataset_identity['orig_text'][i]
                idis = eval(dataset_identity['idis'][i])
                label = dataset_identity['label'][i]

                datas = {'text': [orig_text], 'label': [label]}
                dataset = DataFrame(datas)
                logits, predict_label = self.get_predictions_single(model=model, classifier=classifier, dataset=dataset,
                                                                    tokenizer=tokenizer,
                                                                    padding=padding,
                                                                    max_seq_length=max_seq_length, if_identity=False)
                orig_predict_label = predict_label[0]
                get_idi = False
                contain_i_orig = ID.which_identity(orig_text)
                orig_identities.append(contain_i_orig)
                for idi in idis:
                    datas = {'text': [idi], 'label': [label]}
                    dataset = DataFrame(datas)
                    logits, predict_label = self.get_predictions_single(model=model, classifier=classifier,
                                                                        dataset=dataset, tokenizer=tokenizer,
                                                                        padding=padding,
                                                                        max_seq_length=max_seq_length,
                                                                        if_identity=False)
                    if predict_label[0] != orig_predict_label:
                        if get_idi == False:
                            num_idi += 1
                        # contain_i_orig = ID.which_identity(orig_text)
                        contain_i_idi = ID.which_identity(idi)

                        different_identity = []
                        for identity in contain_i_orig:
                            if identity not in contain_i_idi:
                                different_identity.append(identity)
                        for identity in contain_i_idi:
                            if identity not in contain_i_orig:
                                different_identity.append(identity)

                        one_orig_text_idi.append(different_identity)
                        get_idi = True
                        # break
                diff_i.append(one_orig_text_idi)
                #     all_idi_labels.append(predict_label[0])
                # if sum(all_idi_labels)/float(len(idis)+1) != 1 and sum(all_idi_labels)/float(len(idis)+1) != 0:
                #     num_idi += 1
            individual_metric = num_idi / float(len(dataset_identity['orig_text']))
            print("-->individual metric:{}, num_idi:{}, num_all:{}".format(individual_metric, num_idi,
                                                                           len(dataset_identity['orig_text'])))

            gender_num = 0
            religion_num = 0
            race_num = 0
            for contain_identity in orig_identities:
                for identity in contain_identity:
                    if identity in ID.identity_category['gender']:
                        gender_num += 1
                    elif identity in ID.identity_category['religion']:
                        religion_num += 1
                    elif identity in ID.identity_category['race']:
                        race_num += 1

            gender_num_idi = 0
            religion_num_idi = 0
            race_num_idi = 0
            for one_samples_i in diff_i:
                gender_exist = False
                religion_exist = False
                race_exist = False
                for diff_identity in one_samples_i:  # e.g., diff_identity: ['male', 'homosexual']
                    if all(i in ID.identity_category['gender'] for i in diff_identity) and gender_exist == False:
                        gender_num_idi += 1
                        gender_exist = True
                    elif all(i in ID.identity_category['religion'] for i in diff_identity) and religion_exist == False:
                        religion_num_idi += 1
                        religion_exist = True
                    elif all(i in ID.identity_category['race'] for i in diff_identity) and race_exist == False:
                        race_num_idi += 1
                        race_exist = True

            gender_metric = gender_num_idi / float(gender_num)
            religion_metric = religion_num_idi / float(religion_num)
            race_metric = race_num_idi / float(race_num)
            print("-->individual metric for gender:{}, num_idi:{}, num_all:{}".format(gender_metric, gender_num_idi,
                                                                                      gender_num))
            print(
                "-->individual metric for religion:{}, num_idi:{}, num_all:{}".format(religion_metric, religion_num_idi,
                                                                                      religion_num))
            print("-->individual metric for race:{}, num_idi:{}, num_all:{}".format(race_metric, race_num_idi,
                                                                                    race_num))
            # print("-->diff_i", diff_i)
            # with open("diff_identity_train.txt", "w") as f:
            #     for one_idi_identity in diff_i:
            #         # Write each item to the file, followed by a newline character
            #         f.write(f"{str(one_idi_identity)}\n")
            return individual_metric, gender_metric, religion_metric, race_metric
        elif metric == 'fnr' or metric == 'fpr':
            all_metrics = {'acc': [], 'fpr': [], 'fnr': []}
            # ----------group metric on orig_text----------
            # datas = {'text': dataset_identity['orig_text'], 'label': dataset_identity['label']}
            # dataset = DataFrame(datas)
            # texts = dataset['text'].tolist()
            # base_labels = dataset['label'].tolist()

            # logits, predict_labels = self.get_predictions_single(model=model, classifier=classifier, dataset=dataset,
            #                                                      tokenizer=tokenizer,
            #                                                      padding=padding, max_seq_length=max_seq_length,
            #                                                      if_identity=False)
            print("-->all_length:", len(predict_labels))
            base_labels = dataset_identity['label']
            texts = dataset_identity['text']
            labels_orig = []
            labels_predict = []
            labels_truth = []
            ID = IdentityDetect()
            for j in range(0, len(ID.identities)):
                identity = ID.identities[j]
                select_labels_truth = []
                select_labels_predict = []
                for i in range(0, len(texts)):
                    # text = dataset['text'][i]
                    text = texts[i]
                    text = text.lower()
                    if ID.identity_detect(text, identity) == True:
                        # label = dataset['label'][i]
                        label = base_labels[i]
                        predict_label = predict_labels[i]
                        select_labels_truth.append(label)
                        select_labels_predict.append(predict_label)
                labels_truth.append(select_labels_truth)
                labels_predict.append(select_labels_predict)

            for i in range(0, len(ID.identities)):
                identity = ID.identities[i]
                print("-->identity", identity)
                print("-->length", len(labels_truth[i]))
                TP, FP, TN, FN = self.perf_measure(labels_truth[i], labels_predict[i])
                try:
                    FPR = FP / (FP + TN)
                except:
                    print("(FP + TN) is 0")
                    FPR = None
                try:
                    FNR = FN / (TP + FN)
                except:
                    print("(TP + FN) is 0")
                    FNR = None
                ACC = accuracy_score(labels_truth[i], labels_predict[i])
                print("-->ACC", ACC)
                print("-->F1", f1_score(labels_truth[i], labels_predict[i]))
                all_metrics['acc'].append(ACC)
                all_metrics['fpr'].append(FPR)
                all_metrics['fnr'].append(FNR)
                print("-->FPR", FPR)
                print("-->FNR", FNR)
                # print("-->positive probability:{}, {}/{}".format(
                #     list(labels_predict[i]).count(1) / len(labels_predict[i]),
                #     list(labels_predict[i]).count(1), len(labels_predict[i])))
                # print("-->group-truth label positive probability:{}, {}/{}".format(
                #     list(labels_truth[i]).count(1) / len(labels_truth[i]),
                #     list(labels_truth[i]).count(1), len(labels_truth[i])))

            all_metrics['fpr'] = [x for x in all_metrics['fpr'] if x != None]
            all_metrics['fnr'] = [x for x in all_metrics['fnr'] if x != None]

            fpr_diff = [abs(fpr - baseline_fpr_train) for fpr in all_metrics['fpr']]
            print("-->fpr_diff", fpr_diff)
            print([fpr - baseline_fpr_train for fpr in all_metrics['fpr']])
            fnr_diff = [abs(fnr - baseline_fnr_train) for fnr in all_metrics['fnr']]
            print("-->fnr_diff", fnr_diff)
            print([fnr - baseline_fnr_train for fnr in all_metrics['fnr']])

            fur_diff = fpr_diff
            print("-->gender mean FUR diff", sum(fur_diff[0:3])/float(len(fur_diff[0:3])))
            print("-->religion mean FUR diff", sum(fur_diff[3:6]) / float(len(fur_diff[3:6])))
            # print("-->religion mean FUR diff", sum(fur_diff[3:5]) / float(len(fur_diff[3:5])))
            print("-->race mean FUR diff", sum(fur_diff[6:8]) / float(len(fur_diff[6:8])))
            # print("-->race mean FUR diff", sum(fur_diff[5:7]) / float(len(fur_diff[5:7])))


            print("-->FPR:{}, mean FPR:{}, mean FPR diff:{}".format(all_metrics['fpr'],
                                                                    np.mean(np.array(all_metrics['fpr'])),
                                                                    np.mean(np.array(fpr_diff))))
            print("-->FNR:{}, mean FNR:{}, mean FNR diff:{}".format(all_metrics['fnr'],
                                                                    np.mean(np.array(all_metrics['fnr'])),
                                                                    np.mean(np.array(fnr_diff))))
            return np.mean(np.array(fpr_diff)), np.mean(np.array(fnr_diff)), \
                   np.mean(np.array(all_metrics['fpr'])), np.mean(np.array(all_metrics['fnr'])), \
                   all_metrics['fpr'], all_metrics['fnr']
            return

    def get_idis(self, model, classifier, dataset_name, tokenizer, padding, max_seq_length):
        num_idi = 0
        idi_texts = []
        idi_labels = []
        dataset_identity = pd.read_csv(dataset_name)
        for i in range(len(dataset_identity['orig_text'])):
            orig_text = dataset_identity['orig_text'][i]
            idis = eval(dataset_identity['idis'][i])
            label = dataset_identity['label'][i]
            datas = {'text': [orig_text], 'label': [label]}
            dataset = DataFrame(datas)
            logits, predict_label = self.get_predictions_single(model=model, classifier=classifier, dataset=dataset,
                                                                tokenizer=tokenizer,
                                                                padding=padding,
                                                                max_seq_length=max_seq_length, if_identity=False)
            orig_pred_label = predict_label[0]
            for idi in idis:
                datas = {'text': [idi], 'label': [label]}
                dataset = DataFrame(datas)
                logits, predict_label = self.get_predictions_single(model=model, classifier=classifier, dataset=dataset,
                                                                    tokenizer=tokenizer,
                                                                    padding=padding,
                                                                    max_seq_length=max_seq_length, if_identity=False)
                if predict_label[0] != orig_pred_label:
                    idi_texts.append(idi)
                    idi_labels.append(orig_pred_label)
                    num_idi += 1
                    break
        individual_metric = num_idi / float(len(dataset_identity['orig_text']))
        print("-->individual metric:{}, num_idi:{}, num_all:{}".format(individual_metric, num_idi,
                                                                       len(dataset_identity['orig_text'])))
        datas = {'text': idi_texts, 'label': idi_labels}
        dataset_idis = DataFrame(datas)
        return dataset_idis


    def get_baseline(self, model, dataset, tokenizer, padding, max_seq_length):
        """
        return:
        number of comments for each identity subgroup
        accuracy
        FPR/FNR of all texts
        FPR/FNR for each identity subgroup
        mean subgroup FPR
        subgroup AUC for each identity subgroup
        mean subgroup AUC
        positive probability for each identity subgroup
        """
        base_model = self.CustomModel.base_model
        # model = self.CustomModel.custom_model["custom_module"]
        text = dataset['text']

        all_FPR = []
        all_FNR = []
        all_AUC = []
        for i in range(len(identities)):
            identity = identities[i]
            print("-->identity", identity)

            new_texts, relate_labels = self.get_texts_with_given_identity(dataset, identity)
            print("-->number of text", len(new_texts))

            from pandas.core.frame import DataFrame
            datas = {"text": new_texts,
                 "label": relate_labels}
            dataset = DataFrame(datas)

            # model = None
            logits, labels, identities_result = self.get_predictions(model, dataset, tokenizer, padding, max_seq_length, True)
            predict_labels = labels
            base_labels = dataset['label']
            select_base_labels = np.array(base_labels)
            select_predict_labels = np.array(predict_labels)

            if len(new_texts) == 0:
                break

            print("-->select_base_labels", select_base_labels)
            print("-->select_predict_labels", select_predict_labels)

            cnf_matrix = confusion_matrix(select_base_labels, select_predict_labels)
            print(cnf_matrix)

            FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            TP = np.diag(cnf_matrix)
            TN = cnf_matrix.sum() - (FP + FN + TP)

            FP = FP.astype(float)
            FN = FN.astype(float)
            TP = TP.astype(float)
            TN = TN.astype(float)
            print("-->FP:{}, FN:{}, TP:{}, TN:{}".format(FP, FN, TP, TN))
            # Sensitivity, hit rate, recall, or true positive rate
            TPR = TP / (TP + FN)
            # Specificity or true negative rate
            TNR = TN / (TN + FP)
            # Precision or positive predictive value
            PPV = TP / (TP + FP)
            # Negative predictive value
            NPV = TN / (TN + FN)
            # Fall out or false positive rate
            FPR = FP / (FP + TN)
            # False negative rate
            FNR = FN / (TP + FN)
            # False discovery rate
            FDR = FP / (TP + FP)
            # Overall accuracy
            ACC = (TP + TN) / (TP + FP + FN + TN)
            print("-->ACC", ACC)
            # AUC = roc_auc_score(select_base_labels, select_predict_labels)
            all_FPR.append(FPR)
            all_FNR.append(FNR)
            # all_AUC.append(AUC)
            print("-->FPR", FPR)
            print("-->FNR", FNR)
            # print("-->AUC", AUC)
            print("-->positive probability:{}, {}/{}".format(list(predict_labels).count(1)/len(predict_labels),
                                                             list(predict_labels).count(1), len(predict_labels)))
        print("-->FPR:{}, mean FPR:{}".format(all_FPR, np.mean(np.array(all_FPR))))
        print("-->FNR:{}, mean FNR:{}".format(all_FNR, np.mean(np.array(all_FNR))))
        # print("-->mean AUC", mean(all_AUC))


    def optimize_test(self):

        # 自定义损失函数

        # 1. 继承nn.Mdule
        class My_loss(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                print("-->(x - y)", (x - y))
                print("-->loss", torch.mean(torch.pow((x - y), 2)))
                return torch.mean(torch.pow((x - y), 2))

        # 2. 直接定义函数 ， 不需要维护参数，梯度等信息
        # 注意所有的数学操作需要使用tensor完成。
        def my_mse_loss(x, y):
            return torch.mean(torch.pow((x - y), 2))

        # 3, 如果使用 numpy/scipy的操作  可能使用nn.autograd.function来计算了
        # 要实现forward和backward函数

        # Hyper-parameters 定义迭代次数， 学习率以及模型形状的超参数
        input_size = 1
        output_size = 1
        num_epochs = 60
        learning_rate = 0.001

        # Toy dataset  1. 准备数据集
        x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                            [9.779], [6.182], [7.59], [2.167], [7.042],
                            [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

        y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                            [3.366], [2.596], [2.53], [1.221], [2.827],
                            [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

        # Linear regression model  2. 定义网络结构 y=w*x+b 其中w的size [1,1], b的size[1,]
        model = nn.Linear(input_size, output_size)
        print("-->original grad")
        for name, parameters in model.named_parameters():
            print("grad:", name, ':', parameters.grad)

        # Loss and optimizer 3.定义损失函数， 使用的是最小平方误差函数
        # criterion = nn.MSELoss()
        # 自定义函数1
        criterion = My_loss()

        # 4.定义迭代优化算法， 使用的是随机梯度下降算法
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        loss_dict = []
        # Train the model 5. 迭代训练
        for epoch in range(num_epochs):
            # Convert numpy arrays to torch tensors  5.1 准备tensor的训练数据和标签
            inputs = torch.from_numpy(x_train)
            targets = torch.from_numpy(y_train)

            # Forward pass  5.2 前向传播计算网络结构的输出结果
            outputs = model(inputs)
            # 5.3 计算损失函数
            # loss = criterion(outputs, targets)

            print("-->outputs", outputs)
            print("-->targets", targets)

            # 1. 自定义函数1
            loss = criterion(outputs, targets)
            # 2. 自定义函数
            # loss = my_mse_loss(outputs, targets)
            # Backward and optimize 5.4 反向传播更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 可选 5.5 打印训练信息和保存loss
            loss_dict.append(loss.item())
            if (epoch + 1) % 5 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

            for name, parameters in model.named_parameters():
                print("grad:", name, ':', parameters.grad)


