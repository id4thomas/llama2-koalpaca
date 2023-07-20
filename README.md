# llama2-koalpaca
Training llama2 weights with koalpaca dataset

## Settings
### Dataset
* KoAlpaca dataset v1.1 by Beomi processed for llama2 chat model
	* https://github.com/Beomi/KoAlpaca
	* https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a

### Model, Training
* meta-llama/Llama-2-{}-chat-hf models
* QLora (4bit) Training


## Dataset Preparation
I used the chat_completion data preprocessing code described [here](https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212).
Preprocessing notebook is provided [here](koalpaca_to_llama_chat_input.ipynb)
The following is an example of llama chat model input:
```python
## dialog input example
dialogs = [
    {"role": "system", "content": "Testing system"},
    {"role": "user", "content": "Testing user"},
    {"role": "assistant", "content": "Testing assistant"}
]

## processing dialog
[
{'role': 'user', 'content': '<<SYS>>\nTesting system\n<</SYS>>\n\nTesting user'}, 
{'role': 'assistant', 'content': 'Testing assistant'}
]

## tokenizer input
['[INST] <<SYS>>\nTesting system\n<</SYS>>\n\nTesting user [/INST] Testing assistant ']
```
With the koalpaca dataset the result are as following:
```python

## Dialog Example
[
	{'role': 'user', 'content': '양파는 어떤 식물 부위인가요? 그리고 고구마는 뿌리인가요?'}, 
	{'role': 'assistant', 'content': '양파는 잎이 아닌 식물의 줄기 부분입니다. ...'}
]

## Dialog input mapped to tokenizer input
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

양파는 어떤 식물 부위인가요? 그리고 고구마는 뿌리인가요? [/INST] 양파는 잎이 아닌 식물의 줄기 부분입니다. 고구마는 식물의 뿌리 부분입니다. 

식물의 부위의 구분에 대해 궁금해하는 분이라면 분명 이 질문에 대한 답을 찾고 있을 것입니다. 양파는 잎이 아닌 줄기 부분입니다...</s>
```

## Training
* QLora 4bit training
* Used Parameters
	* r: 8, alpha: 16, dropout 0.05
	* learning_rate: 2e-5
* Used System Prompt: "당신은 도움이 되며, 예의바르고, 솔직한 조수입니다. 항상 안전하면서도 가능한 한 도움이 되는 답변을 해주세요. 아래는 작업을 설명하는 지시사항입니다. 요청을 적절히 완료하는 응답을 작성해주세요. 조수는 사용자의 질문에 도움이 되고, 상세하며, 정중한 답변을 합니다."

Training Code: [train_qlora.py](train_qlora.py)

```
python train_qlora.py
	--out_dir weights \
	--config_dir configs/13b_chat_hf.json \
	--val_ratio 0.05
```