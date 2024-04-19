import os
os.environ["TMPDIR"] = "/tmp"

import numpy as np
import matplotlib.pyplot as plt
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
    BertForSequenceClassification,
    LlamaForSequenceClassification,
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
torch.set_grad_enabled(True)


class CustomModel:
    def __init__(self, base_model, custom_model=None, do_position=None):
        self.base_model = base_model
        self.custom_model = custom_model
        self.do_position = do_position

    def generate_custom_model(self, AddedModule, position=None):
        """
        addedLayerList: the layers which will be inserted
        """
        if self.custom_model != None:
            print("Already have a sub model!!")

        if isinstance(self.base_model, LlamaForSequenceClassification):
            norm = self.base_model.model.norm
            classifier = self.base_model.score
            self.custom_model = {"norm": norm, "custom_module": AddedModule,
                                 "classifier": classifier}
        else:
            oldModuleList = self.base_model.bert.encoder.layer
            layer_length = len(oldModuleList)  # 12
            pooler = self.base_model.bert.pooler
            dropout = self.base_model.dropout
            # todo: add custom layers after encoder and before classifier (dense type; classifier type; dropout type; dense + activation?)
            # e.g.
            # AddedLayerList = nn.ModuleList()
            # hidden_size = 768
            # AddedLayerList.append(nn.Linear(hidden_size, hidden_size))
            # AddedLayerList.append(nn.Tanh())
            classifier = self.base_model.classifier
            # self.sub_model = [pooler, dropout, classifier]
            self.custom_model = {"pooler": pooler, "dropout": dropout, "custom_module": AddedModule,
                                 "classifier": classifier}
        return self

    def get_base_internal_output(self, inputs, position=None):
        """
        inputs: raw texts
        position(default): right before classifier and after Dropout
        RETURN: output of the layer before custom layer (e.g., encoder output by default)
        """
        if position == None:
            # encoder = self.base_model.bert.encoder
            base_outputs = self.base_model(**inputs)
            base_hidden_states = base_outputs.hidden_states

            base_class_output = base_outputs.logits
            attention_mask = inputs['attention_mask']
        # else:
        #     print("-->position", position)
        return base_internal_output


    def get_base_bertmodel_output(self, inputs, position=None):
        import subprocess

        def check_gpu_usage():
            try:
                result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv'])
                print(result.decode('utf-8'))
            except subprocess.CalledProcessError as e:
                print("Error while running nvidia-smi:", e)
        """
        inputs: raw texts
        position(default): right before classifier and after Dropout
        RETURN: output of the layer before custom layer (e.g., encoder output by default)
        """
        if position == None:
            if isinstance(self.base_model, LlamaForSequenceClassification):
                with torch.no_grad():
                    # print("-->self.base_model.model device", self.base_model.model.device)
                    # check_gpu_usage()
                    transformer_outputs = self.base_model.model(**inputs)
                hidden_states = transformer_outputs[0]
                return hidden_states
            else:
                # encoder = self.base_model.bert.encoder
                # base_outputs = self.base_model.bert(**inputs)
                # priont("-->get base_outputs", base_outputs)
                with torch.no_grad():
                    base_outputs = self.base_model.bert(**inputs)
                # base_outputs = self.base_model.bert(**inputs)
                pooled_output = base_outputs[1]
                pooled_output = self.base_model.dropout(pooled_output)
                logits = self.base_model.classifier(pooled_output)
                # print("-->logits", logits)
                return pooled_output

    # additional operations for LlamaModels
    def pooled_logits(self, logits, inputs):
        ### additional operations for LlamaModels
        if inputs['input_ids'] is not None:
            batch_size = inputs['input_ids'].shape[0]
        else:
            batch_size = inputs['inputs_embeds'].shape[0]

        if self.base_model.config.pad_token_id is None and batch_size != 1:
            raise Valueerror("Cannot handle batch sizes >1 if no padding token is defined.")
        if self.base_model.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if inputs is not None:
                sequence_lengths = torch.eq(inputs['input_ids'], self.base_model.config.pad_token_id).long().argmax(-1) - 1
                sequence_lengths = sequence_lengths % inputs['input_ids'].shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        return pooled_logits

        # '''get pooled_logits'''
        # batch_size = 1
        # if self.base_model.config.pad_token_id is None:
        #     sequence_lengths = -1
        # else:
        #     if input is not None:
        #         sequence_lengths = (torch.eq(inputs['input_ids'], self.base_model.config.pad_token_id).long().argmax(-1) - 1).to(logits.device)
        #         sequence_lengths = sequence_lengths % inputs['input_ids'].shape[-1]
        #         sequence_lengths = sequence_lengths.to(logits.device)
        #     else:
        #         sequence_lengths = -1
        # pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        # logits = pooled_logits
        # return logits

    def forward(self, inputs, classifier):
        """
        inputs: raw texts
        RETURN: output of custom model
        """
        base_internal_output = self.get_base_internal_output(inputs)
        output = self.custom_model['custom_module'].forward(base_internal_output)
        # output = base_internal_output
        # for layer in self.custom_model['custom_module']:
        #     output = layer(output)

        # classifier output
        if classifier != None:
            probability = classifier(output).tolist()
        else:
            probability = self.custom_model['classifier'](output).tolist()
        label_out = probability[0].index(max(probability[0]))
        return probability, label_out

    def forward_with_custom_module(self, inputs, custom_module):
        # base_internal_output = self.get_base_internal_output(inputs)
        base_internal_output = self.get_base_bertmodel_output(inputs)
        output = custom_module.forward(base_internal_output)
        # classifier output
        probability = self.base_model.classifier(output).tolist()
        label_out = probability[0].index(max(probability[0]))
        return probability, label_out

    def check_forward(self, inputs):
        print("-->basic output")
        # encoder = self.base_model.bert.encoder
        base_outputs = self.base_model(**inputs)
        base_class_output = base_outputs.logits
        print(base_class_output)

        print("-->base_internal + classifier")
        base_internal_output = self.get_base_internal_output(inputs)
        output = custom_module.forward(base_internal_output)
        probability = self.custom_model['classifier'](output).tolist()
        label_out = probability[0].index(max(probability[0]))
        return probability, label_out