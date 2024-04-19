# Debias on Transformer Based Models 

[comment]: <> (### Non-intrusive Bias Mitigation for Language Models with Statistical Confidence)
## Text classification task

We get the following results including accuracy, F1 and socres of bias degree on the benchmark datasets:

| Dataset  | Accuracy                       | F1 score      | Group Bias Degree | Individual Bias Degree |
|-------|------------------------------|-------------|---------------|---------------|
| WhiteForumHate  | 91.22%                | 0.9506       | 0.0749         | 0.1962| 
| TwitterHate | 91.88%                     |0.9155       | 0.2369        | 0.1414| 
| GabHate  | 87.34%                 | 0.8585 | 0.1485          | 0.3057| 
| RedditHate | 88.80%      | 0.8098| 0.1060          | 0.1021| 
| GabTwitterHate   | 77.93%                | 0.8269 | 0.0876       | 0.2361| 

## dataset
1. Hate speech dataset from a white supremacist forum (WhiteForumHate) --> dataset/hate_sppech_white/
2. Hate speech dataset from Twitter posts (TwitterHate) --> dataset/hate_speech_twitter/
3. Hate speech dataset from Gab posts (GabHate) --> dataset/hate_speech_online/gab/
4. Hate speech dataset form Reddit posts (RedditHate) --> dataset/hate_speech_online/reddit/
5. Hate and offensive speech from Twitter and Gab posts (TwitterGabHate) --> dataset/hate_speech_offensive/

## fine-tune pre-trained model based on given dataset
e.g. fine-tune bert-base-cased pretrained model with TwitterGabHate dataset
```bash
python main_test.py parameters/base_parameters_train.json
```
--> parameters/base_parameters_trian.json for bert-base-cased model: 
```bash
    "model_name_or_path": "bert-base-cased",  # name of pre-trained model
    "train_file": "dataset/hate_speech_offensive/train.csv",
    "validation_file": "dataset/hate_speech_offensive/test.csv",
    "test_file": "dataset/hate_speech_offensive/test.csv",
    "do_train": "True",
    "do_predict": "True",
    "max_seq_length": 128,
    "per_device_train_batch_size": 32,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "output_dir": "./output/hate_speech_offensive32/",
    "overwrite_output_dir": "true"
```
--> parameters/base_parameters_trian.json for llama model: 
```bash
{
    "model_name_or_path": "meta-llama/Llama-2-7b-hf",
    "train_file": "dataset/hate_speech_offensive/train.csv",
    "validation_file": "dataset/hate_speech_offensive/test.csv",
    "test_file": "dataset/hate_speech_offensive/test.csv",
    "cache_dir": "/root/autodl-tmp/mdzhang/tmp",
    "do_train": "True",
    "do_predict": "True",
    "max_seq_length": 128,
    "per_device_train_batch_size": 32,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "output_dir": "./output/test/",
    "overwrite_output_dir": "true"
}
```

## debias fine-tuned model
debias fine-tuned model with debias adapter
```bash
python main_test.py parameters/base_parameters_hate.json parameters/parameters_fine_tuning_ind_hate.json
```
-->parameters/base_parameters_hate.json
```bash
    "model_name_or_path": "./output/hate_speech_white/",  # path of saved model
    "train_file": "dataset/hate_speech_white/train.csv",
    "validation_file": "dataset/hate_speech_white/test.csv",
    "test_file": "dataset/hate_speech_white/test.csv",
    "max_seq_length": 128,
    "per_device_train_batch_size": 32,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "output_dir": "./output/test/",
    "overwrite_output_dir": "true",
    "fp16": "true"
```
-->orig parameters/parameters_fine_tuning_ind_hate.json
```bash
    "epoch": 20,
    "batch_size": 98,
    "train_learning_rate": 0.001,
    "target": "idi",
    "privileged_label": 1,
    "threshold": 0.1,
    "gamma": 1,
    "baseline_metric_ind": 0.141135108,
    "baseline_fpr_train": 0.05918619,
    "baseline_fnr_train": 0.006885197,
    "baseline_metric_group": 0.0387693241,
    "train_data": "dataset/hate_speech_white/train.csv",
    "test_data": "dataset/hate_speech_white/test.csv",
    "train_data_identity": "dataset/hate_speech_white/train_identity.csv",
    "test_data_identity": "dataset/hate_speech_white/test_identity.csv",
    "save_dir": "models/hate_speech_white/"
```

-->parameters/parameters_fine_tuning_ind_hate.json
```bash
    "target": "idi",
    "threshold": 0.1,
    "baseline_metric": "dataset/hate_speech_white/baseline_metric.jsonl",
    "save_dir": "models/hate_speech_white/",
    "gamma": 1,
    "epoch": 20,
    "batch_size": 98,
    "train_learning_rate": 0.001,
```

