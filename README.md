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

식물의 부위의 구분에 대해 궁금해하는 분이라면 분명 이 질문에 대한 답을 찾고 있을 것입니다. 양파는 잎이 아닌 줄기 부분입니다. 고구마는 다른 질문과 답변에서 언급된 것과 같이 뿌리 부분입니다. 따라서, 양파는 식물의 줄기 부분이 되고, 고구마는 식물의 뿌리 부분입니다.

 덧붙이는 답변: 고구마 줄기도 볶아먹을 수 있나요? 

고구마 줄기도 식용으로 볶아먹을 수 있습니다. 하지만 줄기 뿐만 아니라, 잎, 씨, 뿌리까지 모든 부위가 식용으로 활용되기도 합니다. 다만, 한국에서는 일반적으로 뿌리 부분인 고구마를 주로 먹습니다.</s>
```