import numpy as np

import matplotlib.pyplot as plt
# from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# import ElbowVisualizer
# from yellowbrick.cluster import KElbowVisualizer
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("-->device", device)


# def output_hidden_states(model):
#

class Causality:

    def __init__(self, model, do_layer=0, do_neurons=[0], do_values=None, pert_size=0.01):
        """
        model: base model
        do_layer: intervention layer index, e.g.,9: keep layer 10, 11 and 12.
        do_neuron: intervention neuron index
        """

        self.base_model = model
        self.do_layer = do_layer
        self.do_neurons = do_neurons
        # self.do_values = do_values[self.do_neurons[0]]
        self.do_values = do_values
        self.pert_size = pert_size

        self.y_fair_metrics = ["group_fairness", "FPED", "FNED"]
        # self.y_fair_metrics = ["FPED", "FNED"]
        # self.y_fair_metrics = ["group_fairness"]

        self.base_model.to(device)

    # def get_internal_output(self, inputs):
    #     """
    #     inputs: the tokenizer inputs of raw texts
    #     """
    #     base_outputs = self.base_model(**inputs)
    #     base_hidden_states = base_outputs.hidden_states
    #     base_layer_output = base_hidden_states[self.do_layer]
    #     base_class_output = base_outputs.logits
    #     return base_layer_output, base_class_output
    #
    # def generate_sub_model(self):
    #     oldModuleList = self.base_model.bert.encoder.layer
    #     layer_length = len(oldModuleList)  # 12
    #     num_layers_to_keep = list(range(0, layer_length+1))[self.do_layer+1:]
    #     print("-->num_layer_to_keep", num_layers_to_keep)
    #     # num_layers_to_keep = [10, 11, 12]
    #     newModuleList = nn.ModuleList()
    #     for i in num_layers_to_keep:
    #         newModuleList.append(oldModuleList[i - 1])
    #     model = copy.deepcopy(self.base_model)
    #     model.bert.encoder.layer = newModuleList
    #     model.bert.embeddings = None
    #     return model
    #
    # def forward_sub_model(self, sub_model, inputs):
    #     """
    #     inputs: the input of the generated sub model
    #     e.g., base_layer_9_output: bert.encoder's 9th layer's output as the input of sub model
    #     """
    #     new_model = sub_model.bert.encoder
    #     outputs = new_model(inputs)
    #     encoder_outputs = outputs.last_hidden_state
    #
    #     pooler = sub_model.bert.pooler
    #     pooler_outputs = pooler(encoder_outputs)
    #
    #     dropout = sub_model.dropout
    #     dropout_outputs = dropout(pooler_outputs)
    #
    #     classifier = sub_model.classifier
    #     class_outputs = classifier(dropout_outputs)
    #     return class_outputs
    #
    # def perturb(self, layer_output, do_neuron, value):
    #     """
    #     layer_output: output of intervention layer which need to be perturbed
    #     do_value: fixed value of given do_layer's do_neuron
    #     """
    #     perturbed_output = layer_output
    #     # perturbed_output = layer_output.copy()
    #     # print("-->perturbed_output", perturbed_output)
    #     # print("-->value", np.array(value).shape)
    #     perturbed_output[0][do_neuron] = torch.Tensor(value)
    #     return perturbed_output
    #
    #     # all_perturbed_outputs = []
    #     # for value in self.do_values:
    #     #     perturbed_output[do_neuron] = value
    #     #     all_perturbed_outputs.append(perturbed_output)
    #     # return all_perturbed_outputs
    #
    # def intervention(self, inputs):
    #     """
    #     inputs: the tokenizer inputs of raw texts
    #     """
    #     SubModel = SubModel(self.base_model, self.do_layer)
    #     sub_model = SubModel.generate_sub_model()
    #     class_output, label_output = sub_model.forward(inputs)
    #     print("-->label_output", label_output)
    #
    #     # sub_model = self.generate_sub_model()
    #     # base_layer_output, base_class_output = self.get_internal_output(inputs)
    #     #
    #     # class_output = self.forward_sub_model(sub_model, base_layer_output)
    #     # print("-->base logits", base_class_output)
    #     # print("-->new logits", class_output)
    #     # probability = class_output.tolist()[0]
    #     # label_output = probability.index(max(probability))
    #     # print("-->label_output", label_output)
    #     return class_output, label_output

    def get_yfair_do_value(self, raw_texts, tokenizer, padding, max_seq_length, label, typical_term, term_list, do_neuron, do_value):
        Submodel = SubModel(self.base_model, self.do_layer)
        sub_model = Submodel.generate_sub_model()

        # sub_model.to(device)

        # """test sub_model prediction accuracy """
        # predictions = []
        # for text in tqdm(raw_texts):
        #     inputs = tokenizer(text, padding=padding, max_length=max_seq_length, truncation=True,
        #                        return_tensors="pt")
        #     class_output, label_output = sub_model.forward_orig(inputs)
        #     predictions.append(label_output)
        # # print("-->predictions", predictions)
        # accuracy_metric = load_metric('accuracy')
        # print("-->accuracy without intervention:", accuracy_metric.compute(references=label, predictions=predictions))
        #
        # Fairness_metric = FairMetrics(raw_texts, label, predictions, self.base_model, sub_model, tokenizer, padding,
        #                               max_seq_length, typical_term, term_list)
        # # testing_texts, labels, predictions, base_model, model, tokenizer, padding, max_seq_length, term, kind_terms, *args
        # diff_fair_score = []
        # for metric_name in self.y_fair_metrics:
        #     group_fair_score = Fairness_metric.choosen_metric(metric_name)
        #     diff_fair_score.append(group_fair_score)

        predictions = []
        for text in tqdm(raw_texts):
        # for text in tqdm(raw_texts):
            # print("-->text", text)
            inputs = tokenizer(text, padding=padding, max_length=max_seq_length, truncation=True,
                               return_tensors="pt")
            inputs.to(device)
            if self.do_layer > 12:
                class_output, label_output = sub_model.forward_do_pooler(inputs, do_neuron, do_value)
            else:
                class_output, label_output = sub_model.forward_do_neuron(do_neuron, do_value, inputs, self.pert_size)
                # class_output, label_output = sub_model.forward_orig(inputs)
            predictions.append(label_output)

        # print("-->labels", label)
        # print("-->predictions", predictions)
        accuracy_metric = load_metric('accuracy')
        print("-->accuracy after intervention:", accuracy_metric.compute(references=label, predictions=predictions))
        same = 0
        for i in range(0, len(label)):
            if label[i] == predictions[i]:
                same += 1

        Fairness_metric = FairMetrics(raw_texts, label, predictions, self.base_model, sub_model, tokenizer, padding,
                                      max_seq_length, typical_term, term_list)
        # testing_texts, labels, predictions, base_model, model, tokenizer, padding, max_seq_length, term, kind_terms, *args
        diff_fair_score = []
        for metric_name in self.y_fair_metrics:
            group_fair_score = Fairness_metric.choosen_metric(metric_name)
            diff_fair_score.append(group_fair_score)
        return diff_fair_score
        # group_fair_score = Fairness_metric.group_fairness()
        # return group_fair_score

    def get_yfair_all_values(self, raw_texts, tokenizer, padding, max_seq_length, label, model, typical_term, term_list):
        all_yfair = []
        print("-->self.do_neurons", self.do_neurons)
        # if there is only one intervention layer, we can generate sub_model first as a parameters later
        all_neuron_ie = []
        for n in range(0, len(self.do_neurons)):
            do_neuron = self.do_neurons[n]  # [0, ..., 127], [0, ..., 767]
            do_values = self.do_values[do_neuron]
            print("-->do_neuron", do_neuron)

            # running with multi threads
            all_yfair = []

            def process_function(value):
                fair_score = self.get_yfair_do_value(raw_texts, tokenizer, padding, max_seq_length, label, typical_term,
                                                     term_list, do_neuron, value)
                all_yfair.append(fair_score)
                print("-->do_neuron:{}, do_value:{}, fair_score:{}".format(do_neuron, value, fair_score))

            pool = ThreadPool()
            pool.map(process_function, do_values)
            # pool.map(process_function, [do_values[0]])
            pool.close()
            pool.join()
            # process_function(do_values)  # do_values[0]

            all_ie = []
            for i in range(0, len(self.y_fair_metrics)):
                metric = self.y_fair_metrics[i]
                ie = [score_per_value[i] for score_per_value in all_yfair if score_per_value != None]
                print("For metric {}, ie:{}".format(metric, ie))
                all_ie.append(ie)
                # AIE = sum(ie)/len(ie)
                # AIE_for_metrics.append(AIE)
                # print("For fairness metric {}, average yfair ie:{}".format(metric, AIE))
            # return AIE_for_metrics
            all_neuron_ie.append(all_ie)
        return all_neuron_ie



