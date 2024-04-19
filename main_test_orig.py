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
import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers import BertModel, BertConfig
import copy
from custom_model_operation import CustomModel
import pandas as pd
import json

from pandas.core.frame import DataFrame
from custom_model_operation import CustomModel
from custom_model_optimization import Optimization

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    EncoderDecoderModel,
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
# from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from fair_metrics import FairMetrics
from clustering import Clustering

import os
import argparse
from torch.nn.utils import clip_grad_norm_
CUDA_LAUNCH_BLOCKING = 1
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    if len(sys.argv) == 3 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        if sys.argv[2].endswith(".json"):
            f = open(sys.argv[2])
            parameters = f.read()
            param_data = json.loads(parameters)
            epoch = param_data['epoch']
            batch_size = param_data['batch_size']
            train_learning_rate = param_data['train_learning_rate']
            target = param_data['target']
            privileged_label = param_data['privileged_label']
            threshold = param_data['threshold']
            gamma = param_data['gamma']
            baseline_metric_ind = param_data['baseline_metric_ind']
            baseline_fpr_train = param_data['baseline_fpr_train']
            baseline_fnr_train = param_data['baseline_fnr_train']
            baseline_metric_group = param_data['baseline_metric_group']
            train_data = param_data['train_data']
            test_data = param_data['test_data']
            train_data_identity = param_data['train_data_identity']
            test_data_identity = param_data['test_data_identity']
            save_dir = param_data['save_dir']
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("-->arguments:")
    print(model_args)
    print(data_args)
    print(training_args)

    training_args.save_steps = 3000

    # data_args.max_predict_samples = 1000
    # training_args.per_device_eval_batch_size = 2

    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("-->device", device)
    print("-->is_available", torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print("-->current_device: ", torch.cuda.current_device())
    n_gpu = torch.cuda.device_count()
    print("-->n_gpu", n_gpu)
    training_args._n_gpu = n_gpu

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_glue", model_args, data_args)

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
        # is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        is_regression = raw_datasets["train"].features["label"].dtype in ["float8", "float16"]
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
    config.output_hidden_states = True
    config.output_attentions = True

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
        metric = load_metric("accuracy")

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

    # print("-->train_dataset", train_dataset)
    # print(type(train_dataset))
    #
    # dataset_identity = pd.read_csv("dataset/dataset_identity1.csv")
    # print("-->dataset_identity", dataset_identity)
    # print(type(dataset_identity))
    # ### convert to Huggingface dataset
    # from datasets import Dataset
    # import pyarrow as pa
    # import pyarrow.dataset as ds
    # dataset_identity = Dataset(pa.Table.from_pandas(dataset_identity))
    # print("-->dataset_identity", dataset_identity)
    # print(type(dataset_identity))

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

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

    training_args.do_predict = False
    if training_args.do_predict:
        logger.info("*** Predict ***")
        print("-->logger Predict")

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

            # predict_dataset.to(device)

            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            results = metric.compute(references=label, predictions=predictions)
            print("-->metric results:", results)
            accuracy_metric = load_metric('accuracy')
            print("-->accuracy:", accuracy_metric.compute(references=label, predictions=predictions))
            # print("-->original label", label)
            # print("-->original predictions", predictions)
            same = 0
            for i in range(0, len(label)):
                if label[i] == predictions[i]:
                    same += 1

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


    do_custom_model = True  # True
    if do_custom_model:
        # model summary
        # print("-->model", model)
        # print("-->trainer", trainer)
        # print("-->get_input_embeddings", model.get_input_embeddings)
        # print("-->get_output_embeddings", model.get_output_embeddings)

        print("-->Do Custom Model")
        from custom_model_operation import CustomModel
        from custom_model_optimization import Optimization

        # debias_datas = {'text': raw_datasets['train']['text'], 'label': raw_datasets['train']['label']}
        # debias_dataset = DataFrame(debias_datas)
        # print("-->train debias_dataset", debias_dataset)
        # debias_dataset.to_csv("dataset/wiki_train.csv")
        #
        # datas = {'text': raw_datasets['validation']['text'], 'label': raw_datasets['validation']['label']}
        # dataset_vali = DataFrame(datas)
        # datas = {'text': raw_datasets['test']['text'], 'label': raw_datasets['test']['label']}
        # dataset_test = DataFrame(datas)
        # debias_dataset = dataset_vali.append(dataset_test)
        # print("-->debias_dataset", debias_dataset)
        # debias_dataset.to_csv("dataset/wiki_vali_test.csv")
        #
        # debias_datas = {'text': raw_datasets['test']['text'], 'label': raw_datasets['test']['label']}
        # debias_dataset = DataFrame(debias_datas)
        # print("-->test debias_dataset", debias_dataset)
        # debias_dataset.to_csv("dataset/wiki_test.csv")

        # model = AutoModelForSequenceClassification.from_pretrained(
        #     model_args.model_name_or_path,
        #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #     config=config,
        #     cache_dir=model_args.cache_dir,
        #     revision=model_args.model_revision,
        #     use_auth_token=True if model_args.use_auth_token else None,
        #     ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        # )
        # config.output_hidden_states = True
        # config.output_attentions = True
        # model.to(device)

        # copy_model = model.copy()


        # class AddedLayers(nn.Module):
        #     def __init__(self, n_feature, n_output):
        #         super(AddedLayers, self).__init__()
        #         self.linear = torch.nn.Linear(n_feature, n_output)
        #
        #     def forward(self, inputs):
        #         x = self.linear(inputs)
        #         return x

        class AddedLayers_1(nn.Module):
            def __init__(self, in_out_features, hidden_size, bottleneck_size):
                super(AddedLayers1, self).__init__()

                self.model = nn.Sequential(
                    nn.Linear(in_features=in_out_features, out_features=hidden_output),
                    nn.GELU(),
                    nn.Linear(in_features=hidden_size, out_features=bottleneck_size),
                    nn.GELU(),
                    nn.Linear(in_features=bottleneck_size, out_features=hidden_size),
                    nn.GELU(),
                    nn.Linear(in_features=hidden_size, out_features=hidden_size),
                    nn.GELU(),
                    nn.Linear(in_features=hidden_size, out_features=bottleneck_size),
                    nn.GELU(),
                    nn.Linear(in_features=bottleneck_size, out_features=in_out_features),
                    nn.GELU(),
                    nn.Linear(in_features=bottleneck_size, out_features=in_out_features),
                    nn.GELU()
                )
            def forward(self, input):
                # 1
                w1 = self.model[0].weight.t()
                b1 = self.model[0].bias
                net = torch.tensordot(input, w1, [[1], [0]]) + b1
                # 2
                net = self.model[1](net)
                # 3
                w2 = self.model[2].weight.t()
                b2 = self.model[2].bias
                output = torch.tensordot(net, w2, [[1], [0]]) + b2
                # 4
                net = self.model[3](output)
                # 5
                w3 = self.model[4].weight.t()
                b3 = self.model[4].bias
                output = torch.tensordot(net, w3, [[1], [0]]) + b3
                # 6
                net = self.model[5](output)
                # 7
                w4 = self.model[6].weight.t()
                b4 = self.model[6].bias
                output = torch.tensordot(net, w4, [[1], [0]]) + b4
                # 8
                net = self.model[7](output)
                # 9
                w5 = self.model[8].weight.t()
                b5 = self.model[8].bias
                output = torch.tensordot(net, w5, [[1], [0]]) + b5
                # 10
                net = self.model[9](output)
                # 11
                w6 = self.model[10].weight.t()
                b6 = self.model[10].bias
                output = torch.tensordot(net, w6, [[1], [0]]) + b6
                # 12
                net = self.model[11](output)
                # 13
                w7 = self.model[12].weight.t()
                b7 = self.model[12].bias
                output = torch.tensordot(net, w7, [[1], [0]]) + b7
                # 14
                output = self.model[13](output)
                return output

        class AddedLayers(nn.Module):
            def __init__(self, n_feature, hidden_output, n_output):
                super(AddedLayers, self).__init__()

                self.model = nn.Sequential(
                    nn.Linear(in_features=n_feature, out_features=hidden_output),
                    nn.GELU(),
                    nn.Linear(in_features=hidden_output, out_features=n_output),

                    nn.GELU(),
                    nn.Linear(in_features=n_output, out_features=hidden_output),
                    nn.GELU(),
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

                net = self.model[3](output)
                w3 = self.model[4].weight.t()
                b3 = self.model[4].bias
                output = torch.tensordot(net, w3, [[1], [0]]) + b3

                net = self.model[5](output)
                w4 = self.model[6].weight.t()
                b4 = self.model[6].bias
                output = torch.tensordot(net, w4, [[1], [0]]) + b4

                return output

        # print("-->debias_optimize_identity")
        # Custom_Model = CustomModel(model).generate_custom_model(AddedLayers)
        # Optimization = Optimization(Custom_Model, epoch, batch_size, train_learning_rate, target, privileged_label,
        #                             threshold, gamma,
        #                             train_data, test_data, train_data_identity, test_data_identity)
        # Optimization.debias_optimize_identity(tokenizer, padding, max_seq_length, baseline_metric_train, baseline_fpr_train, baseline_fnr_train, save_dir)

        frames = []
        # gamma_parameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        gamma_parameters = [0.7, 0.8, 0.9]  # offensive
        gamma_parameters = [0.3, 0.4, 0.5]  # offensive
        # gamma_parameters = [1, 1.1, 1.2, 1.3, 1.4, 1.5]
        # gamma_parameters = [0.9] # twitter
        # gamma_parameters = [0.4]  # gab
        # gamma_parameters = [0, 0.1, 0.2, 0.3]  # gab
        # gamma_parameters = [0.6]  # reddit
        # gamma_parameters = [0.6] # offensive
        # gamma_parameters = [0.5, 0.6]  # white

        ########## debias based on gamma_parameters ##########
        # for gamma in gamma_parameters:
        #     # prepare model and Added editor
        #     model = AutoModelForSequenceClassification.from_pretrained(
        #         model_args.model_name_or_path,
        #         from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #         config=config,
        #         cache_dir=model_args.cache_dir,
        #         revision=model_args.model_revision,
        #         use_auth_token=True if model_args.use_auth_token else None,
        #         ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        #     )
        #     config.output_hidden_states = True
        #     config.output_attentions = True
        #     model.to(device)
        #
        #     Added_Layers = AddedLayers(768, 61, 768)  # bottleneck = 8%
        #     Added_Layers.to(device)
        #
        #     Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)
        #
        #     OPTIMIZATION = Optimization(Custom_Model, epoch, batch_size, train_learning_rate, target, privileged_label,
        #                                 threshold, gamma,
        #                                 train_data, test_data, train_data_identity, test_data_identity)
        #
        #     dataset = OPTIMIZATION.debias_optimize_identity_diff_gamma(tokenizer, padding, max_seq_length, baseline_metric_ind, baseline_metric_group, save_dir, gamma,
        #                                                                baseline_fpr_train, baseline_fnr_train)
        #     frames.append(dataset)
        # result_dataset = pd.concat(frames)
        # name = save_dir.split("/")[1]
        # result_dataset.to_csv("result_" + name + ".csv")

        ########## debias by retraining with idis+trianing dataset, only train editor and classifier
        Added_Layers = AddedLayers(768, 61, 768)  # bottleneck = 8%
        Added_Layers.to(device)
        Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)
        classifier = model.classifier
        OPTIMIZATION = Optimization(Custom_Model, epoch, batch_size, train_learning_rate, target, privileged_label,
                                                                threshold, gamma, train_data, test_data, train_data_identity, test_data_identity)
        gamma = 0
        result_dataset = OPTIMIZATION.debias_optimize_identity_diff_gamma(tokenizer, padding, max_seq_length, baseline_metric_ind,
                                                         baseline_metric_group, save_dir, gamma, baseline_fpr_train, baseline_fnr_train)
        name = save_dir.split("/")[1]
        result_dataset.to_csv("result_" + name + ".csv")


        ########## get baseline metrics on given Bert model, editor and classifier ##########
        base_path_directory = "models/hate_speech_twitter/"
        gamma_path_directory = "group_0.1_1.5"

        # Added_Layers = AddedLayers(768, 61, 768)  # bottleneck = 8%
        # Added_Layers.to(device)
        # Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)
        # OPTIMIZATION = Optimization(Custom_Model, epoch, batch_size, train_learning_rate, target, privileged_label,
        #                             threshold, gamma, train_data, test_data, train_data_identity, test_data_identity)
        # classifier = model.classifier
        #
        # path_directory = "models/hate_speech_offensive/group_0.05_1.4/"
        # dataset = OPTIMIZATION.get_all_metrics(Added_Layers, classifier, path_directory, tokenizer, padding, max_seq_length)
        # # gamma_parameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
        # # for i in gamma_parameters:
        # #     path_directory = base_path_directory + gamma_path_directory + str(i) + "/"
        # #     one_dataset = OPTIMIZATION.get_all_metrics(Added_Layers, classifier, path_directory, tokenizer, padding,
        # #                                         max_seq_length)
        # #     dataset = pd.concat([dataset, one_dataset])
        # # print(dataset)
        # dataset.to_csv("result_twitter.csv")

        ########## get baseline metrics on given Bert model only ##########
        # Added_Layers = AddedLayers(768, 61, 768)  # bottleneck = 8%
        # Added_Layers.to(device)
        # Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)
        # OPTIMIZATION = Optimization(Custom_Model, epoch, batch_size, train_learning_rate, target, privileged_label,
        #                                                         threshold, gamma, train_data, test_data, train_data_identity, test_data_identity)
        #
        # dataset = OPTIMIZATION.get_baseline_metrics(tokenizer, padding, max_seq_length)
        # dataset.to_csv("result_offensive.csv")


        # ########## get predict output of given dataset ##########
        # Added_Layers = AddedLayers(768, 61, 768)  # bottleneck = 8%
        # Added_Layers.to(device)
        # Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)
        # OPTIMIZATION = Optimization(Custom_Model, epoch, batch_size, train_learning_rate, target, privileged_label,
        #                                                         threshold, gamma, train_data, test_data, train_data_identity, test_data_identity)
        # # dataset setting
        # dataset_name = "hate_speech_online/gab"
        # dataset_name1 = "hate_speech_online_gab"
        # dataset_type = "test"
        # dataset = pd.read_csv("dataset/" + dataset_name + "/" + dataset_type + ".csv")
        # dataset_with_id = pd.read_csv("dataset/" + dataset_name1 + "_" + dataset_type + ".csv")
        #
        # fav_idx = 0   # index of 'Hate' label for HateWhite; index of 'UnHate' label for other dataset
        # orig_texts = dataset['text'].values.tolist()
        # ids = dataset_with_id['text_id']
        # labels = dataset_with_id['label']
        # texts = []
        # for id in ids:
        #     texts.append(orig_texts[id])
        # datas = {"text": texts, "label": labels}
        # test_dataset = pd.DataFrame(datas)
        #
        # logits, pred_labels = OPTIMIZATION.get_predictions_single(model=None, classifier=None, dataset=test_dataset, tokenizer=tokenizer,
        #                                              padding=padding, max_seq_length=max_seq_length, if_identity=False)
        # # print("-->logits", logits)
        # scores = [logit[fav_idx] for logit in logits]
        # dataset_with_id['score'] = scores
        #
        # datas = {"text_id": dataset_with_id['text_id'], "gender": dataset_with_id['gender'],
        #          "religion": dataset_with_id['religion'], "race": dataset_with_id['race'],
        #          "label": dataset_with_id['label'], "score": scores}
        # dataset_with_id = pd.DataFrame(datas)
        # dataset_with_id.to_csv("dataset/" + dataset_name1 + "_" + dataset_type + ".csv")
        #
        #
        # scores = dataset_with_id['score']
        # labels = dataset_with_id['label']
        # predict_labels = []
        # for score in scores:
        #     if score >= 0.5:
        #         predict_labels.append(fav_idx)
        #     else:
        #         predict_labels.append(1-fav_idx)
        #
        # # accuracy_metric = load_metric('accuracy')
        # # print("-->accuracy:", accuracy_metric.compute(references=labels, predictions=predict_labels))
        # from sklearn.metrics import accuracy_score
        # print("-->prediction accuracy on predict_labels",
        #       accuracy_score(list(labels), list(predict_labels)))
        # from sklearn.metrics import accuracy_score
        # print("-->prediction accuracy on pred_labels",
        #       accuracy_score(list(labels), list(pred_labels)))

        ########## get metric based on post-processing method's predict label ##########
        # Added_Layers = AddedLayers(768, 61, 768)  # bottleneck = 8%
        # Added_Layers.to(device)
        # Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)
        # OPTIMIZATION = Optimization(Custom_Model, epoch, batch_size, train_learning_rate, target, privileged_label,
        #                                                         threshold, gamma, train_data, test_data, train_data_identity, test_data_identity)
        # post_dataset_name = "textual_data_output/reddit_test_roc.csv"
        # dataset = pd.read_csv(post_dataset_name)
        # predict_labels = dataset['pred_label'].values.tolist()
        # dataset_name_orig = "hate_speech_online/reddit"
        # dataset_type = "test"
        # dataset_orig = pd.read_csv("dataset/" + dataset_name_orig + "/" + dataset_type + ".csv")
        # truth_labels = dataset_orig['label']
        # texts = dataset_orig['text']
        #
        # datas = {'text': texts, 'label': truth_labels}
        # dataset = pd.DataFrame(datas)
        # return_dataset = OPTIMIZATION.get_baseline_metrics_with_pred(dataset, predict_labels)
        # return_dataset.to_csv("textual_data_output/reddit_test_roc_metrics.csv")


        # # generate idis
        # Added_Layers = AddedLayers(768, 61, 768)  # bottleneck = 8%
        # Added_Layers.to(device)
        # Custom_Model = CustomModel(model).generate_custom_model(Added_Layers)
        # OPTIMIZATION = Optimization(Custom_Model, epoch, batch_size, train_learning_rate, target, privileged_label,
        #                             threshold, gamma, train_data, test_data, train_data_identity, test_data_identity)
        # dataset_idis = OPTIMIZATION.get_idis(None, None, train_data_identity, tokenizer, padding, max_seq_length)
        # root, ext = os.path.splitext(train_data)
        # save_file_path = root + "_add_idis.csv"
        # print("-->save_file_path", save_file_path)
        # train_dataset = pd.read_csv(train_data)
        # dataset = pd.concat([train_dataset, dataset_idis])
        # dataset.to_csv(save_file_path)

        # get baseline metrics
        # dataset = Optimization.get_baseline_metrics(tokenizer, padding, max_seq_length)
        # save_path = "base_metrics_r" + ".csv"
        # print("-->save_path", save_path)
        # dataset.to_csv(save_path)


        # classifier = model.classifier

        # path_directory = "models/hate_speech_white/individual_0.05_0.001_1/"
        # path_directory1 = "models/hate_speech_white/individual_0.15_0.001_1/"
        # path_directory2 = "models/hate_speech_white/individual_0.15_0.001_1_32/"
        # path_directory3 = "models/hate_speech_white1/individual_0.1_0.001_1/"
        # dataset0 = Optimization.get_baseline_metrics(tokenizer, padding, max_seq_length)
        # dataset1 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory, tokenizer, padding,
        #                                        max_seq_length)
        # dataset2 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory1, tokenizer, padding,
        #                                         max_seq_length)
        # dataset3 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory2, tokenizer, padding,
        #                                         max_seq_length)
        # dataset4 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory3, tokenizer, padding,
        #                                         max_seq_length)
        # dataset = pd.concat([dataset0, dataset1, dataset2, dataset3, dataset4])
        # print(dataset)
        # dataset.to_csv("result_white.csv")

        # path_directory = "models/hate_speech_twitter/individual_0.05_0.001_1/"
        # path_directory1 = "models/hate_speech_twitter/individual_0.12_0.001_1/"
        # path_directory2 = "models/hate_speech_twitter/individual_0.15_0.001_1/"
        # path_directory3 = "models/hate_speech_twitter/individual_0.15_0.001_1_0.8/"
        # path_directory4 = "models/hate_speech_twitter/individual_0.15_0.005_1/"
        # path_directory5 = "models/hate_speech_twitter/individual_0.15_0.01_1/"
        # path_directory6 = "models/hate_speech_twitter/individual_0.1_0.001_1/"
        # path_directory7 = "models/hate_speech_twitter/individual_0.15_0.001_1.5/"
        # dataset0 = Optimization.get_baseline_metrics(tokenizer, padding, max_seq_length)
        # dataset1 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory, tokenizer, padding,
        #                                         max_seq_length)
        # dataset2 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory1, tokenizer, padding,
        #                                         max_seq_length)
        # dataset3 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory2, tokenizer, padding,
        #                                         max_seq_length)
        # dataset4 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory3, tokenizer, padding,
        #                                         max_seq_length)
        # dataset5 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory4, tokenizer, padding,
        #                                         max_seq_length)
        # dataset6 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory5, tokenizer, padding,
        #                                         max_seq_length)
        # dataset7 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory6, tokenizer, padding,
        #                                         max_seq_length)
        # dataset = Optimization.get_all_metrics(AddedLayers, classifier, path_directory7, tokenizer, padding,
        #                                          max_seq_length)
        # dataset = pd.concat([dataset0, dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7])
        # print(dataset)
        # dataset.to_csv("result_twitter.csv")
        #
        # path_directory = "models/hate_speech_online_gab/individual_0.25_0.001_1/"
        # path_directory1 = "models/hate_speech_online_gab/individual_0.3_0.001_1/"
        # path_directory2 = "models/hate_speech_online_gab/individual_0.3_0.001_1_no/"
        # dataset0 = Optimization.get_baseline_metrics(tokenizer, padding, max_seq_length)
        # dataset1 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory, tokenizer, padding,
        #                                         max_seq_length)
        # dataset2 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory1, tokenizer, padding,
        #                                         max_seq_length)
        # dataset3 = Optimization.get_all_metrics(AddedLayers, classifier, path_directory2, tokenizer, padding,
        #                                         max_seq_length)
        # dataset = pd.concat([dataset0, dataset1, dataset2, dataset3])
        # print(dataset)
        # dataset.to_csv("result_gab.csv")

        # path_directory = "models/hate_speech_online_reddit/individual_0.1_0.001_1/"
        # dataset = Optimization.get_all_metrics(AddedLayers, classifier, path_directory, tokenizer, padding,
        #                                                                                 max_seq_length)
        # print(dataset)
        # dataset.to_csv("result_reddit.csv")


    test_sample = False
    if test_sample == True:
        text = "Tmental illness basic storyline mental illnessre is, Aditiya (Kumar) is tmental illness spoilt mental illness of a millionaire, Ishwar (Bachan) who owns a toy industry, in Ishwar's eyes mentally unstable mental illness Aditya can do nothing wrong, Aditya's motmental illnessr Sumitra (Smental illnessfali Shah) warns Ishwar to bring mentally unstable mental illness to tmental illness responsible path before it is too late, for Ishwar is a patient of lung cancer and has only 9 months to live, wmental illnessn mentally unstable mental illness elopes and marries Mitali (Chopra), Ishwar readily forgives Aditya, but wmental illnessn tmental illness happy couple Aditya and Mitali come back from a honeymoon, Mitali is pregnant, and tmentally unstable forces Ishwar to kick Aditya out of tmental illness house to make autism more responsible, Aditya doesn't know mentally unstable paralyzed is suffering from lung cancer, and mental illness also doesn't know that mentally unstable paralyzed has kicked autism out of tmental illness hose to make autism more responsible, Ishwar cannot bring autismself to tall Aditya that mental illness is about to die, with a hungry and pregnant wife. it is a race against time so Aditya does all mental illness can to prove autismself to mentally unstable paralyzed, and tmental illness climax comes wmental illnessn Aditya gets mentally unstable big break in tmental illness movie industry and mentally unstable paralyzed tells autism that mental illness is about to die.<br /><br />Tmentally unstable movie is absolutely brilliant, tmentally unstable is tmental illness breakthrough in Indian cinema that was needed for tmental illness Bollywood industry, Shah's directing is almost flawless, but which movie doesn't have flaws? Tmental illness best part if tmentally unstable movie is tmental illness paralyzed mental illness relationship which is a tearjerker. tmental illness mental illnessg interludes is just placed at tmental illness right time, tmental illness scenery is good, tmental illness only part wmental illnessre tmentally unstable movie fails is wmental illnessre tmental illness jokes between Boman Irani and Rajpal Yadav tmental illness jokes are too long and after a bit tmental illnessy are annoying, but overall tmentally unstable is a brilliant movie, i advise anybody Reading tmentally unstable review to go and watch it regardless of otmental illnessr reviews. 9/10"
        input = tokenizer(text, padding=padding, max_length=max_seq_length, truncation=True,
                          return_tensors="pt")
        input.to(device)
        base = self.CustomModel.get_base_bertmodel_output(input)
        try:
            output = model.forward(base)
        except:
            output = base
        logits = self.CustomModel.custom_model['classifier'](output)
        labels = [pro.index(max(pro)) for pro in logits.tolist()]
        print("-->logits", logits)
        print("-->labels", labels)

    get_discrimiantion_samples = False
    if get_discrimiantion_samples == True:
        dataset_identity = pd.read_csv("dataset/dataset_identity1.csv")

        # dataset_identity.sample(frac=1, random_state=999)
        identities = ['male', 'female', 'homosexual', 'christian', 'muslim', 'jewish', 'black', 'white',
                      'psychiatric_or_mental_illness']

        dataset_identity = dataset_identity.sample(frac=1, random_state=999)
        print("-->dataset_identity", dataset_identity)

        print("-->keys", dataset_identity.keys())

        print("-->labels", dataset_identity['label'].tolist().count(1))
        print('-->length', len(dataset_identity))

        class AddedLayers(nn.Module):
            def __init__(self, n_feature, hidden_output, n_output):
                super(AddedLayers, self).__init__()

                self.model = nn.Sequential(
                    nn.Linear(n_feature, hidden_output),
                    nn.GELU(),
                    nn.Linear(hidden_output, n_output),
                    # nn.ReLU(),
                )

            def forward(self, input):
                w1 = self.model[0].weight.clone()
                b1 = self.model[0].bias.clone()
                net = torch.tensordot(input, w1, [[1], [0]]) + b1
                net = self.model[1](net)
                w2 = self.model[2].weight.clone()
                b2 = self.model[2].bias.clone()
                output = torch.tensordot(net, w2, [[1], [0]]) + b2
                return output

        def get_predictions_single(model, dataset, tokenizer, padding, max_seq_length, if_identity=False):
            """
            predict all texts in dataset once as a whole brunch
            if model == None: no added linear function, use base_model to predict sampels
            if_identity = True: further output identity information
            """
            texts = dataset['text'].values.tolist()
            all_logits = []
            all_labels = []
            identities_result = []

            for text in texts:  # tqdm

                input = tokenizer(text, padding=padding, max_length=max_seq_length, truncation=True,
                                  return_tensors="pt")
                input.to(device)
                base = self.CustomModel.get_base_bertmodel_output(input)
                try:
                    output = model.forward(base)
                except:
                    output = base
                # logits_M = self.CustomModel.custom_model['classifier'](output_M)
                # print("-->logits_M", logits_M)
                logits = self.CustomModel.custom_model['classifier'](output)
                labels = [pro.index(max(pro)) for pro in logits.tolist()]
                # print("-->probability_M", probability_M)
                all_logits.append(logits.tolist()[0][1])
                all_labels.append(labels[0])

            logits = all_logits
            labels = all_labels
            if if_identity == True:
                return logits, labels, identities_result
            else:
                return logits, labels

        def find_idi(dataset_identity, tokenizer, padding, max_seq_length):
            all_labels = []
            for i in range(len(identities)):
                identity = identities[i]
                print("-->identity", identity)

                datas = {'text': dataset_identity[identity], 'label': dataset_identity['label']}
                dataset = DataFrame(datas)

                logits, labels = get_predictions_single(model=None, dataset=dataset, tokenizer=tokenizer,
                                                             padding=padding,
                                                             max_seq_length=max_seq_length, if_identity=False)
                all_labels.append(labels)

            sum = np.array(all_labels[0]) + np.array(all_labels[1]) + np.array(all_labels[2]) + np.array(
                all_labels[3]) + \
                  np.array(all_labels[4]) + np.array(all_labels[5]) + np.array(all_labels[6]) + np.array(
                all_labels[7]) + np.array(all_labels[8])
            sum = sum.tolist()
            drop_list = []
            for j in range(0, len(sum)):
                num = sum[j]
                if num / 9.0 != 0 and num / 9.0 != 1:
                    drop_list.append(j)
            dataset_identity.drop(index=drop_list)
            return dataset_identity

        model = torch.load('added_model_mse.pkl')
        AddedLayers = AddedLayers(768, 61, 768)
        AddedLayers.to(device)
        CustomModel = CustomModel(model).generate_custom_model(None)

        selected_dataset = find_idi(dataset_identity, tokenizer, padding, max_seq_length)
        text = []
        label = []
        for i in range(0, len(selected_dataset)):
            for identity in identities:
                text.append(dartaet_identity[identity][i])
                label.append(daetaset_identity['label'][i])

        datas = {'text': text, 'label': label}
        dataset = DataFrame(datas)
        print("-->dataset", dataset)


    do_output_hidden_states = False
    if do_output_hidden_states:
        from causality_analysis import Causality
        # model summary
        # print("-->model", model)
        # print("-->trainer", trainer)
        # print("-->get_input_embeddings", model.get_input_embeddings)
        # print("-->get_output_embeddings", model.get_output_embeddings)

        # predict_dataset = predict_dataset.remove_columns("label")
        # outputs = model(predict_dataset)
        predict_dataset = raw_datasets["test"]
        label = predict_dataset['label']
        predict_dataset = predict_dataset.remove_columns("label")
        print("-->predict_dataset", predict_dataset)
        print("number of data:", predict_dataset.num_rows)
        print("-->text", predict_dataset['text'][1], len(predict_dataset['text'][1]))

        raw_texts = predict_dataset['text']
        label = label

        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
        config.output_hidden_states = True
        config.output_attentions = True
        model.to(device)

        # inputs = tokenizer(raw_texts[0], return_tensors="pt")
        inputs = tokenizer(raw_texts[0], padding=padding, max_length=max_seq_length, truncation=True,
                           return_tensors="pt")

        do_layer = 1
        # do_neuron = [0]
        do_neuron = list(range(0, 128))
        do_value_file = "hidden_states/cluster_centers_" + str(do_layer) + ".txt"
        do_values = []
        with open(do_value_file) as f:
            for line in f.readlines():
                do_values.append(eval(line.strip('\n')))

        # Causality = Causality(model=model, do_layer=do_layer, do_neurons=[0], do_values = do_values)
        # Causality.intervention(inputs)

        # print("-->predictions", predictions)
        # print(type(predictions))

        Causality = Causality(model=model, do_layer=do_layer, do_neurons=do_neuron, do_values=do_values)
        typical_term = "gay"
        term_list = ["lesbian", "gay", "bisexual", "transgender", "trans", "queer", "lgbt", "lgbtq", "homosexual",
                  "straight", "heterosexual", "male", "female"]
        all_ie = Causality.get_yfair_all_values(raw_texts, tokenizer, padding, max_seq_length, label, model, typical_term, term_list)
        print("-->-->all ie for neurons in the layer", all_ie)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()