B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# We use the default prompt in llama git repo
# DEFAULT_SYSTEM_PROMPT = """\
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# Korean default system prompt
DEFAULT_SYSTEM_PROMPT = """
"""


def llama_dialog_preprocess(dialog):
	if dialog[0]["role"] != "system":
		dialog = [
			{
				"role": "system",
				"content": DEFAULT_SYSTEM_PROMPT,
			}
		] + dialog
	dialog = [
		{
			"role": dialog[1]["role"],
			"content": B_SYS
			+ dialog[0]["content"]
			+ E_SYS
			+ dialog[1]["content"],
		}
	] + dialog[2:]
	assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
		[msg["role"] == "assistant" for msg in dialog[1::2]]
	), (
		"model only supports 'system', 'user' and 'assistant' roles, "
		"starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
	)
	return dialog

def map_dialog_to_tokenizer_input(dialog):
	return f"{B_INST} {(dialog[0]['content']).strip()} {E_INST} {(dialog[1]['content']).strip()}"

def map_data_to_lambda_text(x):
	instruction = x["instruction"]
	output = x["output"]

	dialog = [
		# {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
		{"role": "user", "content": instruction},
		{"role": "assistant", "content": output}
	]
	processed_dialog = llama_dialog_preprocess(dialog)
	dialog_input = map_dialog_to_tokenizer_input(processed_dialog)
	return dialog_input
