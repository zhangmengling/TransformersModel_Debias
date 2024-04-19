import os
import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    EvalPrediction
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "mlabonne/llama-2-7b-miniguanaco"
# new_model = "./output/llama-2-7b-hate_speech_offensive"
# model_name = "NousResearch/Llama-2-7b-chat-hf"
model_name = "meta-llama/Llama-2-7b-hf"
################################################################################
# QLoRA parameters
################################################################################
# LoRA attention dimension
lora_r = 64
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layers
lora_dropout = 0.1
################################################################################
# bitsandbytes parameters
################################################################################
# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False
compute_dtype = torch.float16
################################################################################
# TrainingArguments parameters
################################################################################
# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"
# Number of training epochs
num_train_epochs = 1
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False
# Batch size per GPU for training
per_device_train_batch_size = 4
# Batch size per GPU for evaluation
per_device_eval_batch_size = 4
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Optimizer to use
optim = "paged_adamw_32bit"
# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True
# Save checkpoint every X updates steps
save_steps = 25
# Log every X updates steps
logging_steps = 25
################################################################################
# SFT parameters
################################################################################
# Maximum sequence length to use
max_seq_length = 128
# Pack multiple short examples in the same input sequence to increase efficiency
packing = False
# Load the entire model on the GPU 0
device_map = "balanced"


################################################################################
# load dataset
################################################################################
train_file = "dataset/hate_speech_offensive/train.csv"
validation_file = "dataset/hate_speech_offensive/test.csv"
test_file = "dataset/hate_speech_offensive/test.csv"
cache_dir = "/root/autodl-tmp/mdzhang/tmp"
use_auth_token = False
padding = "max_length"

# Loading a dataset from your local files.
# CSV/JSON training and evaluation files are needed.
data_files = {"train": train_file, "validation": validation_file, "test":test_file}
train_extension = train_file.split(".")[-1]
test_extension = test_file.split(".")[-1]
assert (
        test_extension == train_extension
), "`test_file` should have the same extension (csv or json) as `train_file`."
data_files["test"] = test_file
raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=cache_dir,
                use_auth_token=True if use_auth_token else None,
)

# Labels
# Trying to have good defaults here, don't hesitate to tweak to your needs.
# is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
is_regression = raw_datasets["train"].features["label"].dtype in ["float8", "float16"]
if is_regression:
    num_labels = 1
else:
    label_list = raw_datasets["train"].unique("label")
    print("-->label_list", label_list)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)








# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token="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",
    cache_dir=cache_dir,
    trust_remote_code=True,
    resume_download=True
)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'

#bits and byte config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
    token="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",
    cache_dir=cache_dir,
    resume_download=True,
    quantization_config=bnb_config,
)

# for name, param in base_model.named_parameters():
#     if "score" in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# model = PeftModel.from_pretrained(base_model, new_model)
# model = model.merge_and_unload()

# model.push_to_hub(new_model, use_temp_dir=False)
# tokenizer.push_to_hub(new_model, use_temp_dir=False)

# Load tokenizer and model with QLoRA configuration
# compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

max_seq_length = min(max_seq_length, tokenizer.model_max_length)
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
# with training_args.main_process_first(desc="dataset map pre-processing"):
#         raw_datasets = raw_datasets.map(
#             preprocess_function,
#             batched=True,
#             load_from_cache_file=True,
#             desc="Running tokenizer on dataset",
#         )
if "train" not in raw_datasets:
    raise ValueError("--do_train requires a train dataset")
dataset = raw_datasets["train"]
# if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
#     raise ValueError("--do_eval requires a validation dataset")
# eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
# if "test" not in raw_datasets and "test_matched" not in raw_datasets:
#     raise ValueError("--do_predict requires a test dataset")
# predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
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


#TrainingArguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    # max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

pipe = pipeline(task="text-classification", model=base_model, tokenizer=tokenizer, max_length=128)

logging.set_verbosity(logging.CRITICAL)

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

#Initialize the SFTTrainer object
trainer = SFTTrainer(
    model=base_model, # model
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

torch.cuda.empty_cache()
# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)

save_path = "./output/llama_lora/"
trainer.model.save_pretrained(save_path)
trainer.save_model(save_path)













# # Load base model
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map=device_map
# )
# # model.config.use_cache = False
# model.config.pretraining_tp = 1

