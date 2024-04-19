#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
split into two identity: female and male (by generating female-related/male-related textual data based on the original one)
add one layer after encoder and before classifier as debias layer
fine tuning the debias layer
use parameters: e.g., python3 test.py parameters/base_parameters_test.json parameters/parameters_fine_tuning1.json
"""

""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import os
os.environ["TMPDIR"] = "/tmp"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'

import matplotlib.pyplot as plt

import logging
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List
from tqdm import tqdm
import datasets
import numpy as np
from datasets import load_dataset, load_metric
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig
import copy
import pandas as pd
import json
from pandas.core.frame import DataFrame

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaTokenizer,
    EncoderDecoderModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
# from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from trl import SFTTrainer

from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
print("-->root_dir", root_dir)
sys.path.append(str(root_dir))

os.chdir(root_dir)

from public_operation.custom_model_operation import CustomModel
from public_operation.custom_model_optimization import Optimization

from public_operation.clustering import Clustering
from public_operation.debias_editor import AddedLayers, AddedLayers_twice
from public_operation.fair_metrics import FairMetrics

import os
import argparse
from torch.nn.utils import clip_grad_norm_
CUDA_LAUNCH_BLOCKING = 1
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import subprocess

def check_gpu_usage():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv'])
        print(result.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print("Error while running nvidia-smi:", e)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(999)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

@dataclass
class DebiasArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    target: str = field(
        metadata={"help": "Debias target. e.g., group (for group bias); individual (for individual bias)."}
    )
    threshold: float = field(
        metadata={"help": "Expected bias degree metric after debiasing."}
    )
    save_dir: str = field(
        metadata={"help": "Model saving directory, e.g., 'models/hate_speech_white/' for WhiteForumHate"}
    )
    baseline_metrics: Optional[str] = field(
        default=None,
        metadata={"help": "The jsonl file name saving baseline metrics"}
    )
    do_debias: bool = field(
        default=False,
        metadata={"help": "Whether to do debiasing on the given baseline model."}
    )
    do_get_baseline: bool = field(
        default=False,
        metadata={"help": "Whether to get the baseline metrics of the given model only"}
    )
    do_get_baseline_with_editor: bool = field(
        default=False,
        metadata={"help": "Whether to get the baseline metrics based on given model and (debias editor + debias classifier)."}
    )
    gamma: Optional[float] = field(
        default=1.0,
        metadata={"help": "Debias training gamma: weight of bias loss."},
    )
    epoch: int = field(
        default=20,
        metadata={"help": "Debias training epoch."},
    )
    batch_size: int = field(
        default=98,
        metadata={"help": "Debias training batch size."},
    )
    train_learning_rate: float = field(
        default=0.001,
        metadata={"help": "Debias training learning rate"},
    )
    train_data: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."}
    )
    test_data: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."}
    )
    train_data_identity: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."}
    )
    test_data_identity: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."}
    )
    baseline_metric_group: Optional[float] = field(
        default=None,
        metadata={"help": "Baseline group bias degree."}
    )
    baseline_metric_ind: Optional[float] = field(
        default=None,
        metadata={"help": "Baseline individual bias degree."}
    )
    baseline_fpr_train: Optional[float] = field(
        default=None,
        metadata={"help": "Baseline group bias degree."}
    )
    baseline_fnr_train: Optional[float] = field(
        default=None,
        metadata={"help": "Baseline group bias degree."}
    )
    privileged_label: Optional[float] = field(
        default=None,
        metadata={"help": "Baseline group bias degree."}
    )
    debias_path_directory: Optional[str] = field(
        default=None,
        metadata={"help": "Path directory saving debias editor and classifier"}
    )

    def __post_init__(self):
        if self.baseline_metrics != None:
            if os.path.exists(self.baseline_metrics):
                with open(self.baseline_metrics) as json_file:
                    baseline_metrics = json.load(json_file)
                self.baseline_metric_group = baseline_metrics['baseline_metric_group_train']
                self.baseline_metric_ind = baseline_metrics['baseline_metric_ind_train']
                self.baseline_fpr_train = baseline_metrics['baseline_fpr_train']
                self.baseline_fnr_train = baseline_metrics['baseline_fnr_train']
                self.privileged_label = baseline_metrics['privileged_label']
            else:
                raise Exception("Exception: The baseline metrics file does not exist.")
        else:
            print("Need calculate baseline metric first!")

        if self.do_get_baseline_with_editor:
            required_params = [
                "debias_path_directory"
            ]
            missing_params = [param for param in required_params if getattr(self, param) is None]
            if missing_params:
                raise Exception(
                    f"Exception: The following parameters are required when do_get_baseline_with_editor is True: {', '.join(missing_params)}")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    debias_parser = HfArgumentParser((DebiasArguments))
    debias_parameters = False

    if len(sys.argv) == 3 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        debias_args, = debias_parser.parse_json_file(json_file=os.path.abspath(sys.argv[2]))
        if debias_args.train_data == None:
            debias_args.train_data = data_args.train_file
        if debias_args.test_data == None:
            debias_args.test_data = data_args.test_file
        if debias_args.train_data_identity == None:
            debias_args.train_data_identity = debias_args.train_data.split('.csv')[0] + "_identity.csv"
        if debias_args.test_data_identity == None:
            debias_args.test_data_identity = debias_args.test_data.split('.csv')[0] + "_identity.csv"
        print("-->debias_args")
        print(debias_args)
        debias_parameters = True
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        try:
            parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DebiasArguments))
            model_args, data_args, training_args, debias_args = parser.parse_args_into_dataclasses()
            if debias_args.train_data == None:
                debias_args.train_data = data_args.train_file
            if debias_args.test_data == None:
                debias_args.test_data = data_args.test_file
            if debias_args.train_data_identity == None:
                debias_args.train_data_identity = debias_args.train_data.split('.csv')[0] + "_identity.csv"
            if debias_args.test_data_identity == None:
                debias_args.test_data_identity = debias_args.test_data.split('.csv')[0] + "_identity.csv"
            print("-->debias_args")
            print(debias_args)
            debias_parameters = True
        except:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("-->arguments:")
    print(model_args)
    print(data_args)
    print(training_args)

    training_args.save_steps = 3000
    # data_args.max_predict_samples = 1000
    # training_args.per_device_eval_batch_size = 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("-->device", device)
    print("-->is_available", torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print("-->current_device: ", torch.cuda.current_device())
    n_gpu = torch.cuda.device_count()
    print("-->n_gpu", n_gpu)
    training_args._n_gpu = n_gpu

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # raw_datasets = load_dataset("glue", 'cola')
        print("-->raw_datasets", raw_datasets)
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        columns_to_remove = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']

        def process_dataset(dataset, threshold=0.5):
            # Convert the 'toxicity' feature to binary labels
            dataset = dataset.map(lambda x: {'label': int(x['toxicity'] > threshold)})
            # Remove the specified columns from each split in the DatasetDict
            dataset = dataset.remove_columns(columns_to_remove)
            return dataset
        if data_args.dataset_name == 'civil_comments':
            raw_datasets['train'] = process_dataset(raw_datasets['train'])
            raw_datasets['validation'] = process_dataset(raw_datasets['validation'])
            raw_datasets['test'] = process_dataset(raw_datasets['test'])

        print("-->raw_datasets", raw_datasets)
        print(type(raw_datasets))
        print("-->train", raw_datasets['train'])
        print(raw_datasets['train']['text'][:10])
        print(raw_datasets['train']['label'][:10])
        # print(type(raw_datasets['train']))
        # raw_datasets = load_dataset(
        #     data_args.dataset_name,
        #     data_args.dataset_config_name,
        #     cache_dir="/mnt/mengdi/.cache",
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file, "test":data_args.test_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        # is_regression = raw_datasets["train"].features["label"].dtype in ["float8", "float16"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            print("-->label_list", label_list)
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if 'Llama' in model_args.model_name_or_path or 'llama' in model_args.model_name_or_path:
        try:
            config = AutoConfig.from_pretrained(
                "./output/llama_config_tokenizer/",
                token="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",
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                resume_download=True
            )
            # LlamaTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "./output/llama_config_tokenizer/",
                token="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",
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                resume_download=True
            )
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        except:
            config = AutoConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                token="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",
                num_labels=num_labels,
                finetuning_task=data_args.task_name,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                resume_download=True
            )
            # LlamaTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                token="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",
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                resume_download=True
            )
            tokenizer.pad_token = tokenizer.eos_token

        # LlamaForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            # torch_dtype=torch.float16,
            cache_dir=model_args.cache_dir,
            device_map="auto",  # balanced, balanced_low_0
            token="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",
            resume_download=True,
            revision=model_args.model_revision,
            # use_auth_token=True if model_args.use_auth_token else None,
            # ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
        print("-->device", device)
        print("-->model.device", model.device)
        config.output_hidden_states = False
        config.output_attentions = False
        model.config.pad_token_id = tokenizer.pad_token_id

        ##### only fine-tune parts of pre-trained llama model #####
        # for name, param in model.named_parameters():
        #     print(name)
        #     if "mlp" in name or "score" in name or 'model.norm' in name:
        #         try:
        #             param.requires_grad = True
        #             print("True")
        #         except:
        #             print("fail to get True")
        #     else:
        #         param.requires_grad = False
        #         print("False")


    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
        model.to(device)
        config.output_hidden_states = False
        config.output_attentions = False

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
        print("-->sentence1_key, sentence2_key", sentence1_key, sentence2_key)
        # non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        # if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        #     sentence1_key, sentence2_key = "sentence1", "sentence2"
        # else:
        #     if len(non_label_column_names) >= 2:
        #         sentence1_key, sentence2_key = non_label_column_names[:2]
        #     else:
        #         sentence1_key, sentence2_key = non_label_column_names[0], None
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        elif "text" in non_label_column_names:
            sentence1_key, sentence2_key = "text", None
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    print("-->raw_datasets", raw_datasets)
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        print("load_metric")
        metric = load_metric("accuracy")
        print("end load_metric")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        print("-->inputs", inputs)
        print(type(inputs))
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    class TrainerwithLossFunction(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.

            Subclass and override for custom behavior.
            """
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs = model(**inputs)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # custom_loss = ...
        # return custom_loss
        preds = inputs[0] if isinstance(inputs, tuple) else inputs
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        label = p.label_ids

    # def compute_individual_metrics(p: EvalPrediction):
    #     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    #     preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    #     label = p.label_ids


    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    print("-->model.device", model.device)
    print(torch.cuda.device_count())
    # Initialize our Trainer
    # training_args.disable_tqdm = True
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # logger.info("**********peft处理model**********")
    # model = get_peft_model(model, lora_config)
    # print("-->print_trainable_parameters")
    # model.print_trainable_parameters()
    #
    # logger.info("**********初始化训练器**********")
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset if training_args.do_train else None,
    #     eval_dataset=eval_dataset if training_args.do_eval else None,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )

    # Training
    if training_args.do_train:
        logger.info("*** Training ***")
        print("-->logger training")
        checkpoint = None
        # training_args.resume_from_checkpoint = "./output/wiki/checkpoint-100000"
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        print("-->logger evaluate")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        print("-->logger Predict")
        trainer.disable_tqdm = True

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            print("-->type", type(predict_dataset))
            label = predict_dataset['label']
            print("-->predict_dataset", predict_dataset)
            # print("-->label", label)
            # print("sentence:", predict_dataset['sentence'][0])

            predict_dataset = predict_dataset.remove_columns("label")
            print("num of predict_dataset", predict_dataset.num_rows)
            predict_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

            predict_dataloader = DataLoader(predict_dataset, batch_size=8)

            predictions = []
            with torch.no_grad():
                for batch in tqdm(predict_dataloader):
                    batch = {k: v for k, v in batch.items()}

                    outputs = model(**batch)
                    logits = outputs.logits

                    # print("-->logits", logits)

                    if is_regression:
                        pred = logits.squeeze().cpu().numpy()
                        predictions.append(pred)
                    else:
                        pred = torch.argmax(logits, dim=1).numpy()
                        predictions += list(pred)
                        # predictions.append(pred)
                        # print("-->pred", pred)
                        # print(predictions)

            # Flatten predictions if they are in a nested structure

            # Compute accuracy or other metrics
            results = metric.compute(references=label, predictions=predictions)
            print("-->metric results:", results)

            # predictions = []
            # size = 1
            # for i in tqdm(range(0, len(predict_dataset), size)):
            #     # Create a small batch
            #     batch = predict_dataset.select(range(i, min(i + size, len(predict_dataset))))
            #     # Predict the output for the current batch (which has 1 data point)
            #     batch_predictions = trainer.predict(batch, metric_key_prefix="predict").predictions
            #     # Process the predictions as needed and store
            #     predictions.append(batch_predictions)

            # predict_dataset = predict_dataset.select(range(5))
            # print(predict_dataset)
            # predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            # print("-->prediction type", predictions)
            # predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            #
            # results = metric.compute(references=label, predictions=predictions)
            # print("-->metric results:", results)
            # accuracy_metric = load_metric('accuracy')
            # print("-->accuracy:", accuracy_metric.compute(references=label, predictions=predictions))
            # # print("-->original label", label)
            # # print("-->original predictions", predictions)
            # same = 0
            # for i in range(0, len(label)):
            #     if label[i] == predictions[i]:
            #         same += 1
            #
            # output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            # if trainer.is_world_process_zero():
            #     with open(output_predict_file, "w") as writer:
            #         logger.info(f"***** Predict results {task} *****")
            #         writer.write("index\tprediction\n")
            #         for index, item in enumerate(predictions):
            #             if is_regression:
            #                 writer.write(f"{index}\t{item:3.3f}\n")
            #             else:
            #                 item = label_list[item]
            #                 writer.write(f"{index}\t{item}\n")
    if debias_parameters:
        print("-->Do Custom Model")
        print("-->model", model)

        hidden_size = model.config.hidden_size
        bottleneck_size = round(hidden_size * 0.08)  # bottleneck = 8%
        print("hidden_size:{}, bottleneck_size:{}".format(hidden_size, bottleneck_size))

        if debias_args.do_debias:

            # check the difference between dataset and dataset_identity
            Added_Layers = AddedLayers(hidden_size, bottleneck_size, hidden_size)  # AddedLayers
            Added_Layers.to(device)

            Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)

            OPTIMIZATION = Optimization(Custom_Model,
                                        debias_args.epoch,
                                        debias_args.batch_size,
                                        debias_args.train_learning_rate,
                                        debias_args.target,
                                        debias_args.privileged_label,
                                        debias_args.threshold,
                                        0.0,  # set gamma as 0.0 temperately
                                        debias_args.train_data,
                                        debias_args.test_data,
                                        debias_args.train_data_identity,
                                        debias_args.test_data_identity,
                                        # torch_dtype=torch.float16
                                        )

            # dataset_train = OPTIMIZATION.train_data.sample(frac=1, random_state=999)
            # dataset_test = OPTIMIZATION.test_data.sample(frac=1, random_state=999)
            # dataset_identity_train = OPTIMIZATION.train_data_identity.sample(frac=1, random_state=999)
            # dataset_identity_test = OPTIMIZATION.test_data_identity.sample(frac=1, random_state=999)
            #
            # print("-->dataset_train")
            # label = dataset_train['label'].tolist()
            # print(label.count(1), label.count(0), label.count(0)/label.count(1))
            # print("-->dataset_test")
            # label = dataset_test['label'].tolist()
            # print(label.count(1), label.count(0), label.count(0)/label.count(1))
            # print("-->dataset_identity_train")
            # label = dataset_identity_train['label'].tolist()
            # print(label.count(1), label.count(0), label.count(0)/label.count(1))
            # print("-->dataset_identity_test")
            # label = dataset_identity_test['label'].tolist()
            # print(label.count(1), label.count(0), label.count(0)/label.count(1))

            print("-->debias_args.do_debias")
            if isinstance(debias_args.gamma, list):
                gamma_parameters = debias_args.gamma   # gamma_parameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            elif isinstance(debias_args.gamma, float):
                gamma_parameters = [debias_args.gamma]  # 1.0
            else:
                raise Exception("Exception: gamma type not correct")

            ########## optimize on accuracy only ##########
            Added_Layers = AddedLayers(hidden_size, bottleneck_size, hidden_size)  # AddedLayers, AddedLayers_twice
            Added_Layers.to(device)

            Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)

            OPTIMIZATION = Optimization(Custom_Model,
                                        debias_args.epoch,
                                        debias_args.batch_size,
                                        debias_args.train_learning_rate,
                                        debias_args.target,
                                        debias_args.privileged_label,
                                        debias_args.threshold,
                                        0.0,   # set gamma as 0.0 temperately
                                        debias_args.train_data,
                                        debias_args.test_data,
                                        debias_args.train_data_identity,
                                        debias_args.test_data_identity,
                                        # torch_dtype=torch.float16
                                        )

            # editor_orig, classifier_orig = OPTIMIZATION.optimize_acc(tokenizer, padding, max_seq_length, debias_args.save_dir)
            # OPTIMIZATION.get_metrics_with_editor(editor_orig, classifier_orig, tokenizer, padding, max_seq_length)

            # # load editor and classifier from path_direct
            # editor_orig = OPTIMIZATION.CustomModel.custom_model["custom_module"]
            # classifier_orig = OPTIMIZATION.CustomModel.custom_model["classifier"]
            #
            # path_directory = debias_args.save_dir + "acc/"
            # state_dict = torch.load(path_directory + "editor.pth")
            # editor_orig.load_state_dict(state_dict)
            # state_dict = torch.load(path_directory + "classifier.pth")
            # classifier_orig.load_state_dict(state_dict)
            #
            # orig_weight = editor_orig.model[0].weight.clone()

            frames = []
            ########## debias based on gamma_parameters ##########

            for gamma in gamma_parameters:
                print("-->gamma", gamma)
                for name, param in model.named_parameters():
                    if "score" not in name:
                        param.requires_grad = False

                # load editor and classifier from path_direct
                editor_orig = OPTIMIZATION.CustomModel.custom_model["custom_module"]
                classifier_orig = OPTIMIZATION.CustomModel.custom_model["classifier"]

                path_directory = debias_args.save_dir + "acc/"
                state_dict = torch.load(path_directory + "editor.pth")
                editor_orig.load_state_dict(state_dict)
                state_dict = torch.load(path_directory + "classifier.pth")
                classifier_orig.load_state_dict(state_dict)

                # new_weight = editor_orig.model[0].weight.clone()
                # if torch.equal(new_weight, orig_weight) == False:
                #     print("model weight not equal!")

                dataset = OPTIMIZATION.debias_optimize_identity_diff_gamma(tokenizer,
                                                                           padding,
                                                                           max_seq_length,
                                                                           debias_args.baseline_metric_ind,
                                                                           debias_args.baseline_metric_group,
                                                                           debias_args.save_dir,
                                                                           gamma,
                                                                           debias_args.baseline_fpr_train,
                                                                           debias_args.baseline_fnr_train,
                                                                           editor_orig, classifier_orig)  # editor_orig, classifier_orig
                frames.append(dataset)
            result_dataset = pd.concat(frames)
            name = debias_args.save_dir.split("/")[1]
            result_dataset.to_csv("results/debias_" + name + ".csv")

        ########## get baseline metrics on given Bert model only ##########
        if debias_args.do_get_baseline==True:
            print("-->do_get_baseline")
            Added_Layers = AddedLayers(hidden_size, bottleneck_size, hidden_size)
            Added_Layers.to(device)
            Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)
            OPTIMIZATION = Optimization(Custom_Model,
                                        debias_args.epoch,
                                        debias_args.batch_size,
                                        debias_args.train_learning_rate,
                                        debias_args.target,
                                        debias_args.privileged_label,
                                        debias_args.threshold,
                                        debias_args.gamma,
                                        debias_args.train_data,
                                        debias_args.test_data,
                                        debias_args.train_data_identity,
                                        debias_args.test_data_identity)

            dataset = OPTIMIZATION.get_baseline_metrics(tokenizer, padding, max_seq_length)
            extracted_str = os.path.basename(os.path.normpath(model_args.model_name_or_path))
            dataset.to_csv("results/baseline_" + str(extracted_str) + "_results.csv")

        ########## get baseline metrics on given Bert model, editor and classifier ##########
        if debias_args.do_get_baseline_with_editor:
            Added_Layers = AddedLayers(hidden_size, bottleneck_size, hidden_size)
            Added_Layers.to(device)
            Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)
            OPTIMIZATION = Optimization(Custom_Model,
                                        debias_args.epoch,
                                        debias_args.batch_size,
                                        debias_args.train_learning_rate,
                                        debias_args.target,
                                        debias_args.privileged_label,
                                        debias_args.threshold,
                                        debias_args.gamma,
                                        debias_args.train_data,
                                        debias_args.test_data,
                                        debias_args.train_data_identity,
                                        debias_args.test_data_identity)
            # classifier = model.classifier
            classifier = Custom_Model.custom_model['classifier']

            path_directory = "models/hate_speech_white_llama/individual_0.05_"
            # dataset = OPTIMIZATION.get_all_metrics(Added_Layers, classifier, debias.debias_path_directory, tokenizer, padding,
            #                                        max_seq_length)
            # Initialize an empty DataFrame
            combined_dataset = pd.DataFrame()
            gamma_parameters = [0.3, 0.4]
            for i in gamma_parameters:
                path = path_directory + str(i) + "/"
                one_dataset = OPTIMIZATION.get_all_metrics(Added_Layers, classifier, path, tokenizer, padding, max_seq_length)
                combined_dataset = pd.concat([combined_dataset, one_dataset])
            print(combined_dataset)
            combined_dataset.to_csv("models/hate_speech_white_llama/results_ind.csv")

        ########### get predict output of given dataset ##########
        get_predict_output = False
        if get_predict_output:
            Added_Layers = AddedLayers(hidden_size, bottleneck_size, hidden_size)
            Added_Layers.to(device)
            Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)
            OPTIMIZATION = Optimization(Custom_Model,
                                        debias_args.epoch,
                                        debias_args.batch_size,
                                        debias_args.train_learning_rate,
                                        debias_args.target,
                                        debias_args.privileged_label,
                                        debias_args.threshold,
                                        debias_args.gamma,
                                        debias_args.train_data,
                                        debias_args.test_data,
                                        debias_args.train_data_identity,
                                        debias_args.test_data_identity)
            # dataset setting
            dataset_name = "hate_speech_online/gab"
            dataset_name1 = "hate_speech_online_gab"
            dataset_type = "test"
            dataset = pd.read_csv("dataset/" + dataset_name + "/" + dataset_type + ".csv")
            dataset_with_id = pd.read_csv("dataset/" + dataset_name1 + "_" + dataset_type + ".csv")

            fav_idx = 0  # index of 'Hate' label for HateWhite; index of 'UnHate' label for other dataset
            orig_texts = dataset['text'].values.tolist()
            ids = dataset_with_id['text_id']
            labels = dataset_with_id['label']
            texts = []
            for id in ids:
                texts.append(orig_texts[id])
            datas = {"text": texts, "label": labels}
            test_dataset = pd.DataFrame(datas)

            logits, pred_labels = OPTIMIZATION.get_predictions_single(model=None, classifier=None, dataset=test_dataset,
                                                                      tokenizer=tokenizer,
                                                                      padding=padding, max_seq_length=max_seq_length,
                                                                      if_identity=False)
            # print("-->logits", logits)
            scores = [logit[fav_idx] for logit in logits]
            dataset_with_id['score'] = scores

            datas = {"text_id": dataset_with_id['text_id'], "gender": dataset_with_id['gender'],
                     "religion": dataset_with_id['religion'], "race": dataset_with_id['race'],
                     "label": dataset_with_id['label'], "score": scores}
            dataset_with_id = pd.DataFrame(datas)
            dataset_with_id.to_csv("dataset/" + dataset_name1 + "_" + dataset_type + ".csv")

            scores = dataset_with_id['score']
            labels = dataset_with_id['label']
            predict_labels = []
            for score in scores:
                if score >= 0.5:
                    predict_labels.append(fav_idx)
                else:
                    predict_labels.append(1 - fav_idx)

            # accuracy_metric = load_metric('accuracy')
            # print("-->accuracy:", accuracy_metric.compute(references=labels, predictions=predict_labels))
            from sklearn.metrics import accuracy_score
            print("-->prediction accuracy on predict_labels",
                  accuracy_score(list(labels), list(predict_labels)))
            from sklearn.metrics import accuracy_score
            print("-->prediction accuracy on pred_labels",
                  accuracy_score(list(labels), list(pred_labels)))

        ########## get metric based on post-processing method's predict label ##########
        get_post_processing_predict_label = False
        if get_post_processing_predict_label:
            Added_Layers = AddedLayers(hidden_size, bottleneck_size, hidden_size)
            Added_Layers.to(device)
            Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)
            OPTIMIZATION = Optimization(Custom_Model,
                                        debias_args.epoch,
                                        debias_args.batch_size,
                                        debias_args.train_learning_rate,
                                        debias_args.target,
                                        debias_args.privileged_label,
                                        debias_args.threshold,
                                        debias_args.gamma,
                                        debias_args.train_data,
                                        debias_args.test_data,
                                        debias_args.train_data_identity,
                                        debias_args.test_data_identity)
            post_dataset_name = "textual_data_output/reddit_test_roc.csv"
            dataset = pd.read_csv(post_dataset_name)
            predict_labels = dataset['pred_label'].values.tolist()
            dataset_name_orig = "hate_speech_online/reddit"
            dataset_type = "test"
            dataset_orig = pd.read_csv("dataset/" + dataset_name_orig + "/" + dataset_type + ".csv")
            truth_labels = dataset_orig['label']
            texts = dataset_orig['text']

            datas = {'text': texts, 'label': truth_labels}
            dataset = pd.DataFrame(datas)
            return_dataset = OPTIMIZATION.get_baseline_metrics_with_pred(dataset, predict_labels)
            return_dataset.to_csv("textual_data_output/reddit_test_roc_metrics.csv")

        generate_idis = False
        if generate_idis:
            Added_Layers = AddedLayers(hidden_size, bottleneck_size, hidden_size)
            Added_Layers.to(device)
            Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)
            OPTIMIZATION = Optimization(Custom_Model,
                                        debias_args.epoch,
                                        debias_args.batch_size,
                                        debias_args.train_learning_rate,
                                        debias_args.target,
                                        debias_args.privileged_label,
                                        debias_args.threshold,
                                        debias_args.gamma,
                                        debias_args.train_data,
                                        debias_args.test_data,
                                        debias_args.train_data_identity,
                                        debias_args.test_data_identity)
            dataset_idis = OPTIMIZATION.get_idis(None, None, train_data_identity, tokenizer, padding, max_seq_length)
            root, ext = os.path.splitext(train_data)
            save_file_path = root + "_add_idis.csv"
            print("-->save_file_path", save_file_path)
            train_dataset = pd.read_csv(train_data)
            dataset = pd.concat([train_dataset, dataset_idis])
            dataset.to_csv(save_file_path)

        analyze_all_metrics = False
        if analyze_all_metrics:
            Added_Layers = AddedLayers(hidden_size, bottleneck_size, hidden_size)
            Added_Layers.to(device)
            Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)
            OPTIMIZATION = Optimization(Custom_Model,
                                        debias_args.epoch,
                                        debias_args.batch_size,
                                        debias_args.train_learning_rate,
                                        debias_args.target,
                                        debias_args.privileged_label,
                                        debias_args.threshold,
                                        debias_args.gamma,
                                        debias_args.train_data,
                                        debias_args.test_data,
                                        debias_args.train_data_identity,
                                        debias_args.test_data_identity)
            classifier = model.classifier

            path_directory = "models/hate_speech_white/individual_0.05_0.001_1/"
            path_directory1 = "models/hate_speech_white/individual_0.15_0.001_1/"
            path_directory2 = "models/hate_speech_white/individual_0.15_0.001_1_32/"
            path_directory3 = "models/hate_speech_white1/individual_0.1_0.001_1/"
            dataset0 = Optimization.get_baseline_metrics(tokenizer, padding, max_seq_length)
            dataset1 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory, tokenizer, padding,
                                                    max_seq_length)
            dataset2 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory1, tokenizer, padding,
                                                    max_seq_length)
            dataset3 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory2, tokenizer, padding,
                                                    max_seq_length)
            dataset4 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory3, tokenizer, padding,
                                                    max_seq_length)
            dataset = pd.concat([dataset0, dataset1, dataset2, dataset3, dataset4])
            print(dataset)
            dataset.to_csv("result_white.csv")

            path_directory = "models/hate_speech_twitter/individual_0.05_0.001_1/"
            path_directory1 = "models/hate_speech_twitter/individual_0.12_0.001_1/"
            path_directory2 = "models/hate_speech_twitter/individual_0.15_0.001_1/"
            path_directory3 = "models/hate_speech_twitter/individual_0.15_0.001_1_0.8/"
            path_directory4 = "models/hate_speech_twitter/individual_0.15_0.005_1/"
            path_directory5 = "models/hate_speech_twitter/individual_0.15_0.01_1/"
            path_directory6 = "models/hate_speech_twitter/individual_0.1_0.001_1/"
            path_directory7 = "models/hate_speech_twitter/individual_0.15_0.001_1.5/"
            dataset0 = Optimization.get_baseline_metrics(tokenizer, padding, max_seq_length)
            dataset1 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory, tokenizer, padding,
                                                    max_seq_length)
            dataset2 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory1, tokenizer, padding,
                                                    max_seq_length)
            dataset3 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory2, tokenizer, padding,
                                                    max_seq_length)
            dataset4 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory3, tokenizer, padding,
                                                    max_seq_length)
            dataset5 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory4, tokenizer, padding,
                                                    max_seq_length)
            dataset6 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory5, tokenizer, padding,
                                                    max_seq_length)
            dataset7 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory6, tokenizer, padding,
                                                    max_seq_length)
            dataset = Optimization.get_all_metrics(AddedLayers, classifier, path_directory7, tokenizer, padding,
                                                   max_seq_length)
            dataset = pd.concat([dataset0, dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7])
            print(dataset)
            dataset.to_csv("result_twitter.csv")

            path_directory = "models/hate_speech_online_gab/individual_0.25_0.001_1/"
            path_directory1 = "models/hate_speech_online_gab/individual_0.3_0.001_1/"
            path_directory2 = "models/hate_speech_online_gab/individual_0.3_0.001_1_no/"
            dataset0 = Optimization.get_baseline_metrics(tokenizer, padding, max_seq_length)
            dataset1 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory, tokenizer, padding,
                                                    max_seq_length)
            dataset2 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory1, tokenizer, padding,
                                                    max_seq_length)
            dataset3 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory2, tokenizer, padding,
                                                    max_seq_length)
            dataset = pd.concat([dataset0, dataset1, dataset2, dataset3])
            print(dataset)
            dataset.to_csv("result_gab.csv")

            path_directory = "models/hate_speech_online_reddit/individual_0.1_0.001_1/"
            dataset = Optimization.get_all_metrics(AddedLayers, classifier, path_directory, tokenizer, padding,
                                                   max_seq_length)
            print(dataset)
            dataset.to_csv("result_reddit.csv")



# def _mp_fn(index):
#     # For xla_spawn (TPUs)
#     main()

if __name__ == "__main__":
    main()

