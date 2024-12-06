import json
from transformers import AutoTokenizer
	
model_name = "./Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

vocab = tokenizer.get_vocab()

with open("llama_vocab_3_1_8B.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)
    
