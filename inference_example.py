import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, TaskType
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from utils.input_utils import DEFAULT_SYSTEM_PROMPT, B_INST, E_INST, B_SYS, E_SYS

def load_kbit_pretrained_model(pretrained_model_name, gpu_num = 0):
	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_use_double_quant=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.bfloat16
	)
	model = AutoModelForCausalLM.from_pretrained(
		pretrained_model_name, 
		quantization_config=bnb_config,
		device_map={"": f"cuda:{gpu_num}"}, # FOR GPU 0
		use_auth_token = HF_AUTH_TOKEN,
		pretraining_tp = 1 # NEEDED OR mat1 mat2 shape err - see llama hf documentation
	)
	return model

def set_lora_weight(model, lora_weight_dir):
	set_peft_model_state_dict(model,torch.load(lora_weight_dir))

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--config_dir', type=str, required=True)
args = parser.parse_args()

# Load Configs
with open("auth_tokens.json", "r") as f:
	HF_AUTH_TOKEN = json.loads(f.read)["hf_token"]\

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

gpu_num = 0
device = torch.device('cuda:' + str(gpu_num)) if torch.cuda.is_available() else torch.device('cpu')


# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
	config["pretrained_model"],
	use_auth_token = HF_AUTH_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token

### Load Model
# Load Pretrained Model
model = load_kbit_pretrained_model(config["pretrained_model"], gpu_num = gpu_num)
model = get_pretrained_model_for_kbit_training(model)

# Load Peft Model
peft_config = get_lora_config(config, inference_mode = True)
peft_model = get_lora_model(model, peft_config)

set_lora_weight(peft_model, os.path.join(out_dir, "model.pt"))
peft_model.eval()

## Generation
def make_model_input(instruction, system_prompt = None):
	dialog = [{"role": "system", "content": system_prompt}] if system_prompt else list()
	dialog += [
		{"role": "user", "content": instruction}
	]
    dialog = llama_dialog_preprocess(dialog)
	return dialog

def map_dialog_to_tokenizer_input(dialog):
	return f"{B_INST} {(dialog[0]['content']).strip()} {E_INST}"

def generate(model, tokenizer, instruction, decode_params):
	dialog = make_model_input(instruction)
	model_input = tokenizer.bos_token + map_dialog_to_tokenizer_input(dialog)

	encoded = tokenizer(
		model_input, 
		return_tensors='pt', 
		return_token_type_ids=False
	)

	gened = model.generate(
		**encoded, 
		eos_token_id=tokenizer.eos_token_id,
		**decode_params
	)
	only_gen = gened[0][encoded["input_ids"].shape[-1]:]
	return tokenizer.decode(only_gen, skip_special_tokens=True)

decode_params = {
	max_new_tokens=512,
	early_stopping=True,
	do_sample=True,
	temperature = 0.9
}

generated = generate(
	peft_model,
	tokenizer,
	'가위 바위 보 게임은 뭔가요?',
	decode_params = decode_params
)

print(generated)