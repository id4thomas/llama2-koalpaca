import numpy as np

def get_pct_below_threshold(lens, threshold):
	lens = np.array(lens)
	len_below = lens[lens<threshold]
	pct = len_below.shape[0]/lens.shape[0]
	return pct

def get_below_threshold_idxs(lens, threshold):
	lens = np.array(lens)
	return np.where(lens<threshold)[0]

def get_token_lens(tokenizer, fulltexts):
	return [tokenizer(fulltext, return_tensors="pt")["input_ids"].shape[-1] for fulltext in fulltexts]