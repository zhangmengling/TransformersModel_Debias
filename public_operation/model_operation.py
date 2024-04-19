import numpy as np

import matplotlib.pyplot as plt
# from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.metrics import davies_bouldin_score

import scipy.cluster.hierarchy as shc
from matplotlib import pyplot
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
random.seed(10)

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

class SubModel:
    def __init__(self, base_model, do_layer, sub_model=None):
        self.base_model = base_model
        self.do_layer = do_layer
        self.sub_model = sub_model

    def generate_sub_model(self):
        if self.sub_model != None:
            print("Already have a sub model!!")
        oldModuleList = self.base_model.bert.encoder.layer
        layer_length = len(oldModuleList)  # 12
        if self.do_layer <= layer_length:
            num_layers_to_keep = list(range(0, layer_length + 1))[self.do_layer + 1:]
            print("-->num_layers_to_keep", num_layers_to_keep)
            newModuleList = nn.ModuleList()
            for i in num_layers_to_keep:
                newModuleList.append(oldModuleList[i - 1])
            sub_model = copy.deepcopy(self.base_model)
            sub_model.bert.encoder.layer = newModuleList
            # print("-->sub_model.bert.encoder", sub_model.bert.encoder)
            # sub_model.bert.encoder = self.base_model.bert.encoder
            sub_model.bert.embeddings = None
            self.sub_model = sub_model
        else:
            pooler = self.base_model.bert.pooler
            dropout = self.base_model.dropout
            classifier = self.base_model.classifier
            self.sub_model = [pooler, dropout, classifier]
        return self

    def get_base_internal_output(self, inputs):
        """
        inputs: the tokenizer inputs of raw texts
        """
        # print("-->base_internal_output inputs:", inputs)
        attention_mask = inputs['attention_mask']
        # print("-->inputs", inputs)
        # base_outputs, base_pooler_output = self.base_model(**inputs)
        base_outputs = self.base_model(**inputs)
        print("-->base_model prediction", inputs, base_outputs)
        # print("-->base_outputs.hidden_states.shape", base_outputs.hidden_states.shape)
        base_hidden_states = base_outputs.hidden_states
        length = len(base_hidden_states)
        if self.do_layer > length:
            base_class_output = base_outputs.logits
            return base_pooler_output, base_class_output, attention_mask
        else:
            base_layer_output = base_hidden_states[self.do_layer]
            base_class_output = base_outputs.logits
            print("-->base_class_output", base_class_output)
            base_embedding_output = base_hidden_states[0]
            # print("-->base_embedding_output", base_embedding_output)
            # print("-->base_layer_output", base_layer_output)
            # print("-->base encoder output", base_hidden_states[12])
            # return base_embedding_output, base_class_output, base_attentions
            return base_layer_output, base_class_output, attention_mask


    def forward_internal_layer(self, internal_inputs, attention_mask):
        """
        internal_inputs: the output of certain internal layer of base_model
        """

        # internal_attention = torch.Tensor([[1]*128])
        # l = [0]*11 + [-10000]*117
        # l = [1]*10 + [0]*118
        # attention_mask = torch.Tensor([l])
        # print("-->attention_mask", attention_mask)

        # output, pooler_output = self.sub_model(input_ids=None, attention_mask=attention_mask, inputs_embeds=internal_inputs)
        output = self.sub_model(input_ids=None, attention_mask=attention_mask, inputs_embeds=internal_inputs)

        # internal_attention = torch.Tensor([1]*128)
        # output = self.sub_model(input_ids=None, attention_mask=internal_attentions, inputs_embeds=internal_inputs)
        # internal_inputs = {'input_ids': internal_inputs[0]}
        # output = self.sub_model(**internal_inputs)
        # output = self.sub_model(input_ids=None, attention_mask=internal_attention, inputs_embeds=x)

        # internal_inputs = {'input_ids': internal_inputs}
        # output = self.sub_model(**internal_inputs)

        # print("-->encoder output", output.hidden_states[-1])
        # print("-->class_output", output.logits)

        class_output = output.logits.tolist()
        label_output = class_output[0].index(max(class_output[0]))

        print("-->class_output:{}, label_output:{}".format(class_output, label_output))

        # return class_output, label_output

        """
        new_model = self.sub_model.bert.encoder
        # l = [0]*11 + [-10000]*117
        # extended_attention_mask = torch.Tensor([[[l]]])
        outputs = new_model(internal_inputs, attention_mask=attention_mask)
        encoder_outputs = outputs.last_hidden_state

        print("-->encoder output 2", encoder_outputs)

        pooler = self.sub_model.bert.pooler
        pooler_outputs = pooler(encoder_outputs)

        dropout = self.sub_model.dropout
        dropout_outputs = dropout(pooler_outputs)

        classifier = self.sub_model.classifier
        # l = classifier.state_dict()
        # print("-->weight&bias:")
        # print(l['weight'])
        # print(l['bias'])
        probability = classifier(dropout_outputs)
        # label_out = probability.index(max(probability))
        # print("-->probability:{}, label_out:{}".format(probability, label_out))
        print("-->probability:{}".format(probability))
        """

        return class_output, label_output

    def forward_internal_layer_split(self, internal_inputs, attention_mask):
        """
        get sub_model's output step by step (pooler-->dropout-->classifier)
        """
        pooler = self.sub_model[0]
        dropout = self.sub_model[1]
        classifier = self.sub_model[2]

        dropout_output = dropout(internal_inputs)
        probability = classifier(dropout_output).tolist()
        label_out = probability[0].index(max(probability[0]))
        # print("-->probability:{}, label_out:{}".format(probability, label_out))

        return probability, label_out



    def forward_orig(self, inputs):
        """
        inputs: the tokenizer inputs of raw texts
        class_output = sub model predicted logits
        label_output = sub model predicted label
        return: the prediction output based on original internal output + sub model output
                it should be equal to the base_model prediction output
        """
        base_layer_output, base_class_output, attention_mask = self.get_base_internal_output(inputs)
        class_output, label_output = self.forward_internal_layer(base_layer_output, attention_mask)
        return class_output, label_output

    def perturb(self, layer_output, do_neuron, value, perturb_size):
        """
        layer_output: output of intervention layer which need to be perturbed
        do_value: fixed value of given do_layer's do_neuron
        e.g.
        base_layer_output, base_class_output = sub_model.get_base_internal_output(inputs)
        perturbed_layer_outputs = self.perturb(base_layer_output, self.do_neurons[0], do_value)
        """
        # print("-->perturb_size", perturb_size)
        # perturb_size = 0.1  # 0.02, 0.05, 0.1
        # print("-->layer_output", layer_output)
        # print("-->layer_output.data", layer_output.data)
        # print(layer_output.data.requires_grad)
        # layer_output.data[0][do_neuron] = torch.Tensor(layer_output[0][do_neuron])
        neuron_value = layer_output[0][do_neuron].tolist()
        # print("-->neuron_value", neuron_value)
        for i in range(0, len(neuron_value)):
            dirct = random.choice([-1, 1])
            neuron_value[i] = neuron_value[i] + dirct * perturb_size
        layer_output.data[0][do_neuron] = torch.Tensor(neuron_value)

        # layer_output.data[0][do_neuron] = torch.Tensor(value)
        # print("-->perturbed_output", layer_output)
        return layer_output

        # # perturbed_output = layer_output
        # perturbed_output = layer_output.clone()
        # print("-->layer_output", perturbed_output)
        # print("grad_fn:", perturbed_output.grad_fn)
        # # perturbed_output[0][do_neuron] = torch.Tensor(value)
        # perturbed_output[0][do_neuron] = torch.Tensor(perturbed_output[0][do_neuron], requires_grad=True)
        # # perturbed_output.grad_fn.data.copy_(layer_output.grad_fn.data)
        # # perturbed_output.grad_fn = layer_output.grad_fn
        # print("-->perturbed_output", perturbed_output)
        # print("grad_fn:", perturbed_output.grad_fn)
        # return perturbed_output

    def intervention(self, layer_output, do_neuron, value):
        layer_value = layer_output[0]
        layer_value[do_neuron] = value
        layer_output[0] = torch.Tensor(layer_value)
        # layer_output[0][do_neuron] = torch.Tensor(value)
        return layer_output

    # neuron_value = layer_output[0][do_neuron].tolist()
    # # print("-->neuron_value", neuron_value)
    # for i in range(0, len(neuron_value)):
    #     dirct = random.choice([-1, 1])
    #     neuron_value[i] = neuron_value[i] + dirct * perturb_size
    # layer_output.data[0][do_neuron] = torch.Tensor(neuron_value)

    def forward_do_neuron(self, do_neuron, do_value, inputs, pert_size):
        """
        do_neuron: index of perturbed neuron
        value: perturb value of given neuron
        inputs: the tokenizer inputs of raw texts
        """
        base_layer_output, base_class_output, attention_mask = self.get_base_internal_output(inputs)
        # perturbed_layer_outputs = self.perturb(base_layer_output, do_neuron, do_value, pert_size)
        perturbed_layer_outputs = base_layer_output
        class_output, label_output = self.forward_internal_layer(perturbed_layer_outputs, attention_mask)

        # base_layer_output, base_class_output = self.get_base_internal_output(inputs)
        # class_output, label_output = self.forward_internal_layer(base_layer_output)

        return class_output, label_output

    def forward_do_pooler(self, inputs, do_neuron, do_value):
        base_pooled_output, base_class_output, attention_mask = self.get_base_internal_output(inputs)
        # print("-->base_class_output", base_class_output)
        perturbed_layer_outputs = self.intervention(base_pooled_output, do_neuron, do_value)
        # perturbed_layer_outputs = base_pooled_output
        class_output, label_output = self.forward_internal_layer_split(perturbed_layer_outputs, attention_mask)
        # print("-->class_output", class_output, label_output)

        return class_output, label_output
