import numpy as np
import matplotlib.pyplot as plt


import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric
import torch.nn.functional as F
from public_operation.model_operation import SubModel

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("-->device", device)

class FairMetrics:
    def __init__(self, testing_texts, labels, predictions, base_model, model, tokenizer, padding, max_seq_length, term, kind_terms, *args):
        self.testing_texts = testing_texts
        self.predictions = predictions
        self.labels = labels
        self.base_model = base_model
        self.model = model
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_seq_length = max_seq_length
        self.term = term
        self.kind_terms = kind_terms

        print("-->label length", len(self.labels))

        Perform = Performance(self.labels, self.predictions)
        self.TP = Perform.TP
        self.FP = Perform.FP
        self.FN = Perform.FN
        self.TN = Perform.TN
        self.FPR = Perform.FPR
        self.FNR = Perform.FNR

    def statistical_parity_differnece(self, term, kind_term):
        """
        P[C = 1|A = 0] - P[C = 1|A = 1]
        """
        score = 0
        return score

    def abs_odds_diff(self):
        """
        1/2 (|P[C = 1|A = 0; Y = 0] - P[C = 1|A = 1; Y = 0]| +
        |P[C = 1|A = 0; Y = 1] - P[C = 1|A = 1; Y = 1]|)
        """
        score = 0
        return score

    def equal_opportunity_diff(self):
        """
        P[C = 1|A = 0; Y = 1] - P[C = 1|A = 1; Y = 1]
        """
        score = 0
        return score

    def disparate_impact(self):
        """
        P[C = 1|A = 0] / P[C = 1|A = 1]
        """

    def FP_equality_diff(self, term):
        """
        Õt ∈T |FPR − FPRt |
        """
        indexs = []
        selected_labels = []
        selected_predictions = []
        for i in range(0, len(self.testing_texts)):
            text = self.testing_texts[i]
            if term in text.lower():
                indexs.append(i)
                selected_labels.append(self.labels[i])
                selected_predictions.append(self.predictions[i])

        if len(indexs) == 0:
            return False

        Perform_term = Performance(selected_labels, selected_predictions)
        FPR_term = Perform_term.FPR

        if FPR_term == False:
            return False

        score = abs(self.FPR - FPR_term)
        print("term {}'s FPR equality diff:{}".format(term, score))
        return score

    def FN_equality_diff(self, term):
        """
        Õt ∈T|FNR − FNRt |
        """
        indexs = []
        selected_labels = []
        selected_predictions = []
        for i in range(0, len(self.testing_texts)):
            text = self.testing_texts[i]
            if term in text:
                indexs.append(i)
                selected_labels.append(self.labels[i])
                selected_predictions.append(self.predictions[i])

        if len(indexs) == 0:
            return False

        Perform_term = Performance(selected_labels, selected_predictions)
        FNR_term = Perform_term.FNR

        if FNR_term == False:
            return False

        score = abs(self.FNR - FNR_term)
        print("term {}'s FNR equality diff:{}".format(term, score))
        return score

    def overall_equality_diff(self):
        FPED = 0
        FNED = 0
        for term in self.kind_terms:
            fpr = self.FP_equality_diff(term)
            if fpr != False:
                FPED += fpr
            fnr = self.FN_equality_diff(term)
            if fnr != False:
                FNED += fnr
        print("FPED:{}; FNED:{}".format(FPED, FNED))
        return FPED, FNED

    def FP_equality_diff_all(self):
        all_score = []
        for term in self.kind_terms:
            score = self.FP_equality_diff(term)
            print("term {}'s FPR equality diff:{}".format(term, score))
            all_score.append(score)
        return all_score

    def FN_equality_diff_all(self):
        all_score = []
        for term in self.kind_terms:
            score = self.FN_equality_diff(term)
            print("term {}'s FNR equality diff:{}".format(term, score))
            all_score.append(score)
        return all_score


    def group_fairness(self):
        """
        term_list: list of terms containing sensitive terms
        e.g., term_list = gender_terms
        raw_texts: raw textual data
        predictions: model predicted labels
        """
        print("-->logger testing fairness")

        term_list = self.kind_terms
        raw_texts = self.testing_texts
        predictions = self.predictions
        contain_gender = []
        for term in term_list:
            contains = []
            for text in raw_texts:
                if term in text.lower():
                    contains.append(1)
                else:
                    contains.append(0)
            contain_gender.append(contains)

        pos_probabilities = []
        for j in range(0, len(contain_gender)):
            selected_predictions = []
            for i in range(0, len(contain_gender[j])):
                if contain_gender[j][i] == 1:
                    selected_predictions.append(predictions[i])
            try:
                print("-->term {}'s positive proportion of {} is {}:".format(term_list[j], len(selected_predictions),
                                                                             selected_predictions.count(1) / len(
                                                                                 selected_predictions)))
                pos_probabilities.append(selected_predictions.count(1) / len(selected_predictions))

            except:
                print("none text contain term {}".format(term_list[j]))

        baseline = list(predictions).count(1) / len(predictions)
        # print("-->baseline positive proportion:", baseline)
        """test"""
        all_score = []
        for i in range(0, len(pos_probabilities)):
            term = term_list[i]
            score = abs(baseline - pos_probabilities[i])
            print("-->term {}'s ie is {}".format(term, score))
            all_score.append(score)
        # return baseline - pos_probabilities[1]
        return all_score

    def individual_base_fairness(self):
        def perturb(text, term_list):
            inputs = self.tokenizer(text, padding=self.padding, max_length=self.max_seq_length, truncation=True,
                               return_tensors="pt")
            inputs.to(device)
            outputs = self.base_model(**inputs)
            predict_prop = F.softmax(outputs[0], dim=-1).tolist()
            orig_label = predict_prop[0].index(max(predict_prop[0]))
            for term in term_list:
                if term in text.lower():
                    obtain_term = term
                    break
            for term in term_list:
                pert_text = text.lower().replace(obtain_term, term)
                inputs = self.tokenizer(pert_text, padding=self.padding, max_length=self.max_seq_length, truncation=True,
                                   return_tensors="pt")
                inputs.to(device)
                outputs = self.base_model(**inputs)
                predict_prop = F.softmax(outputs[0], dim=-1).tolist()
                # print("-->predict_prop", predict_prop)
                pert_label = predict_prop[0].index(max(predict_prop[0]))
                if pert_label != orig_label:
                    # print("original label:{}, perturbed_label:{}".format(orig_label, pert_label))
                    return True
            return False

        contains = []
        selected_texts = []
        for text in self.testing_texts:
            if any(term in text.lower() for term in self.kind_terms):
                contains.append(1)
                selected_texts.append(text)
            else:
                contains.append(0)

        idi = 0
        for text in selected_texts:
            if perturb(text, self.kind_terms) == True:
                idi += 1
        score = idi / len(selected_texts)
        print("Base CDS:", score)
        return score

    def individual_fairness(self):
        """
        model: original model
        tokenizer: original tokenizer
        term_list: list of terms containing sensitive terms
        e.g., term_list = gender_terms
        raw_texts: raw textual data
        predictions: model predicted labels
        """
        def perturb(text, term_list):
            inputs = self.tokenizer(text, padding=self.padding, max_length=self.max_seq_length, truncation=True,
                               return_tensors="pt")
            inputs.to(device)
            try:
                outputs = self.model(**inputs)
                predict_prop = F.softmax(outputs[0], dim=-1).tolist()
                orig_label = predict_prop[0].index(max(predict_prop[0]))
                # print("-->orig_label", orig_label)
            except:
                class_output, label_output = self.model.forward_orig(inputs)
                orig_label = label_output
            for term in term_list:
                if term in text.lower():
                    obtain_term = term
                    break
            for term in term_list:
                pert_text = text.lower().replace(obtain_term, term)
                inputs = self.tokenizer(pert_text, padding=self.padding, max_length=self.max_seq_length, truncation=True,
                                   return_tensors="pt")
                inputs.to(device)
                try:
                    outputs = self.model(**inputs)
                    predict_prop = F.softmax(outputs[0], dim=-1).tolist()
                    # print("-->predict_prop", predict_prop)
                    pert_label = predict_prop[0].index(max(predict_prop[0]))
                except:
                    class_output, label_output = self.model.forward_orig(inputs)
                    pert_label = label_output
                # print("-->pert_label", pert_label)
                if pert_label != orig_label:
                    # print("original label:{}, perturbed_label:{}".format(orig_label, pert_label))
                    return True
            return False

        base_CDS = self.individual_base_fairness()

        contains = []
        selected_texts = []
        for text in self.testing_texts:
            if any(term in text.lower() for term in self.kind_terms):
                contains.append(1)
                selected_texts.append(text)
            else:
                contains.append(0)

        idi = 0
        for text in selected_texts:
            if perturb(text, self.kind_terms) == True:
                idi += 1
        score = idi / len(selected_texts)
        print("CDS:", score)
        return score

    def choosen_metric(self, metric_name):
        if metric_name == "group_fairness":

            print("Testing group fairness")
            return self.group_fairness()
        elif metric_name == "individual_fairness":
            print("Testing individual fairness")
            return self.individual_fairness()
        elif metric_name == "FPED":
            print("Testing false positive equal opportunity difference")
            return self.FP_equality_diff_all()
        elif metric_name == "FNED":
            print("Testing false negative equal opportunity difference")
            return self.FN_equality_diff_all()


class Performance:
    """
    定义一个类，用来分类器的性能度量
    """

    def __init__(self, labels, predictions):
        """
        :param labels:数组类型，真实的标签
        :param scores:数组类型，分类器的得分
        :param threshold:检测阈值
        """
        self.labels = list(map(int, labels))
        self.predictions = list(map(int, predictions))
        # self.db = self.get_db()
        self.TP, self.FP, self.FN, self.TN = self.get_confusion_matrix()
        # print("-->all performance", self.TP, self.FP, self.FN, self.TN)
        try:
            self.FPR = self.FP / (self.FP + self.TN)
        except:
            print("-->all performance", self.TP, self.FP, self.FN, self.TN)
            self.FPR = False
        try:
            self.FNR = self.FN / (self.FN + self.TP)
        except:
            print("-->all performance", self.TP, self.FP, self.FN, self.TN)
            self.FNR = False

    def accuracy(self):
        """
        :return: 正确率
        """
        return (self.TP + self.TN) / (self.TP + self.FN + self.FP + self.TN)

    def presision(self):
        """
        :return: 准确率
        """
        return self.TP / (self.TP + self.FP)

    def recall(self):
        """
        :return: 召回率
        """
        return self.TP / (self.TP + self.FN)

    def auc(self):
        """
        :return: auc值
        """
        auc = 0.
        prev_x = 0
        xy_arr = self.roc_coord()
        for x, y in xy_arr:
            if x != prev_x:
                auc += (x - prev_x) * y
                prev_x = x
        return auc

    def roc_coord(self):
        """
        :return: roc坐标
        """
        xy_arr = []
        tp, fp = 0., 0.
        neg = self.TN + self.FP
        pos = self.TP + self.FN
        for i in range(len(self.db)):
            tp += self.db[i][0]
            fp += 1 - self.db[i][0]
            xy_arr.append([fp / neg, tp / pos])
        return xy_arr

    def roc_plot(self):
        """
        画roc曲线
        :return:
        """
        auc = self.auc()
        xy_arr = self.roc_coord()
        x = [_v[0] for _v in xy_arr]
        y = [_v[1] for _v in xy_arr]
        plt.title("ROC curve (AUC = %.4f)" % auc)
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.plot(x, y)
        plt.show()

    # def get_db(self):
    #     db = []
    #     for i in range(len(self.labels)):
    #         db.append([self.labels[i], self.scores[i]])
    #     db = sorted(db, key=lambda x: x[1], reverse=True)
    #     return db

    def get_confusion_matrix(self):
        """
        计算混淆矩阵
        :return:
        """
        tp, fp, fn, tn = 0., 0., 0., 0.
        for i in range(len(self.labels)):
            if self.labels[i] == 1 and self.predictions[i] == 1:
                tp += 1
            elif self.labels[i] == 0 and self.predictions[i] == 1:
                fp += 1
            elif self.labels[i] == 1 and self.predictions[i] == 0:
                fn += 1
            else:
                tn += 1
        # print("-->", [tp, fp, fn, tn])
        return tp, fp, fn, tn

