# Using peft 0.4.0 Package
from peft import LoraConfig
from peft import get_peft_model, get_peft_model_state_dict, TaskType

def print_trainable_parameters(model):
	"""
	Prints the number of trainable parameters in the model.
	"""
	trainable_params = 0
	all_param = 0
	for _, param in model.named_parameters():
		all_param += param.numel()
		if param.requires_grad:
			trainable_params += param.numel()
	print(
		f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
	)

def get_pretrained_model_for_kbit_training(model):
	try:
		from peft import prepare_model_for_kbit_training
	except ImportError as error:
		raise NotImplementedError("prepare_model_for_kbit_training is available for peft>=0.4.0")
	except Exception as e:
		raise e

	model.gradient_checkpointing_enable()
	model = prepare_model_for_kbit_training(model)
	return model

def get_lora_config(config):
	return LoraConfig(
			task_type=TaskType.CAUSAL_LM, 
			inference_mode=False, 
			r = config["lora_r"], 
			lora_alpha = config["lora_alpha"], 
			lora_dropout = config["lora_dropout"],
			bias="none",
			# target_modules=["q_proj", "v_proj"]
		)

def get_lora_save_param_dict(model, save_embedding = False):
	state_dict = model.state_dict()
	params_to_save = get_peft_model_state_dict(model, state_dict=state_dict)
	
	if save_embedding:
		layer_keys = list(state_dict.keys())
		embed_keys = list(filter(lambda x: "embed_in" in x, layer_keys))
		for k in embed_keys:
			params_to_save[k] = state_dict[k]
			
	return params_to_save