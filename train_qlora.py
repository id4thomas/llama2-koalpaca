# Modified based on https://colab.research.google.com/gist/Beomi/f163a6c04a869d18ee1a025b6d33e6d8/2023_05_26_bnb_4bit_koalpaca_v1_1a_on_polyglot_ko_12_8b.ipynb
import os
import json
import argparse
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset


with open("auth_tokens.json", "r") as f:
	HF_AUTH_TOKEN = json.loads(f.read)["hf_token"]

# Import Utils
from utils.input_utils import  map_data_to_lambda_text
from utils.data_utils import get_token_lens, get_below_threshold_idxs
from utils.train_utils import set_seed
from utils.peft_utils import get_lora_config, get_lora_model, get_lora_save_param_dict
from utils.peft_utils import get_pretrained_model_for_kbit_training, print_trainable_parameters


def load_kbit_pretrained_model(pretrained_model_name):
	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_use_double_quant=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.bfloat16
	)
	model = AutoModelForCausalLM.from_pretrained(
		pretrained_model_name, 
		quantization_config=bnb_config,
		device_map={"":0}, # FOR GPU 0
		use_auth_token = HF_AUTH_TOKEN,
		pretraining_tp = 1 # NEEDED OR mat1 mat2 shape err - see llama hf documentation
	)
	return model

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--config_dir', type=str, required=True)
parser.add_argument('--val_ratio', type=float, default=0.01)
parser.add_argument('--train_token_cutoff_threshold', type=int, default=1024)
args = parser.parse_args()

# Train Configs
with open(args.config_dir, "r") as f:
	config = json.loads(f.read())

# Prepare
set_seed(config["seed"])
run_name = "{}_lora_r{}_alpha_{}_ep{}_lr{}".format(
	config["pretrained_model"], 
	config["lora_r"],
	config["lora_alpha"],
	config["epochs"], 
	config["learning_rate"]
)

out_dir = os.path.join(args.out_dir, run_name)


## Load Data
data = load_dataset("beomi/KoAlpaca-v1.1a")

tokenizer = AutoTokenizer.from_pretrained(
	config["pretrained_model"],
	use_auth_token = HF_AUTH_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token

# Check BOS, EOS TOKEN
print("BOS", tokenizer.bos_token, tokenizer.bos_token_id)
print("EOS", tokenizer.eos_token, tokenizer.eos_token_id)

# Data Mapping
data = data.map(
    lambda x: {'text': tokenizer.bos_token+map_data_to_lambda_text(x)+tokenizer.eos_token}
)

train_val_split = data["train"].train_test_split(0.01)
train_ds = train_val_split["train"]
val_ds = train_val_split["test"]

train_ds_token_lens = get_token_lens(tokenizer, train_ds["text"])
train_ds_selected_indicies = get_below_threshold_idxs(train_ds_token_lens, args.train_token_cutoff_threshold)
train_ds = train_ds.select(train_ds_selected_indicies)
print("New Train Size", len(train_ds["text"]))

val_ds_token_lens = get_token_lens(tokenizer, val_ds["text"])
val_ds_selected_indicies = get_below_threshold_idxs(val_ds_token_lens, args.train_token_cutoff_threshold)
val_ds = val_ds.select(val_ds_selected_indicies)
print("New Val Size", len(val_ds["text"]))

train_ds = train_ds.map(lambda samples: tokenizer(samples["text"], add_special_tokens=False), batched=True)
val_ds = val_ds.map(lambda samples: tokenizer(samples["text"], add_special_tokens=False), batched=True)

### Load Model
# Load Pretrained Model
model = load_kbit_pretrained_model(config["pretrained_model"])
model = get_pretrained_model_for_kbit_training(model)

# Load Peft Model
peft_config = get_lora_config(config)
peft_model = get_lora_model(model, peft_config)
print_trainable_parameters(peft_model)

## Prepare 
training_args = TrainingArguments(
	run_name = run_name,

	# Train Params
	## Steps/Epochs
	num_train_epochs = config["epochs"],

	## LR
	learning_rate = config["learning_rate"],
	## Batch
	per_device_train_batch_size = config["per_device_batch_size"],
	per_device_eval_batch_size = config["per_device_batch_size"],
	gradient_accumulation_steps = config["gradient_accumulation_steps"],

	# Checkpointing, Saving
	output_dir = os.path.join(out_dir,"checkpoints"),
	save_strategy = "no",
	overwrite_output_dir=True,

	# Evaluating
	evaluation_strategy = "steps",

	# Logging
	logging_dir = out_dir,
	logging_steps = config["summary_step"],
	disable_tqdm = False,
	report_to = ["tensorboard"],

	# System
	seed = config["seed"],
	fp16 = config["fp16"],
	bf16 = config["bf16"],

	# Optimizer
	optim = "paged_adamw_8bit" # "adamw_hf", "adamw_8bit"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
	model = peft_model,
	args = training_args,
	data_collator=data_collator,
	train_dataset=train_ds,
	eval_dataset=val_ds
)

try:
	trainer.train()
except Exception as e:
	raise e

# Final Save DIR
final_out_dir = os.path.join(out_dir, "final")
if not os.path.exists(final_out_dir):
	os.makedirs(final_out_dir, exist_ok = True)

lora_save_dict = get_lora_save_param_dict(trainer.model, save_embedding = False)
torch.save(lora_save_dict, os.path.join(final_out_dir, "model.pt"))

tokenizer.save_pretrained(str(final_out_dir))
with open(os.path.join(final_out_dir, "train_config.json"), 'w') as f:
	json.dump(config, f, indent = 4, ensure_ascii = False)