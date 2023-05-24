from transformers import AutoModelForCausalLM, GPT2Tokenizer

model_name = "gpt2-medium"

# Load pre-trained GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)