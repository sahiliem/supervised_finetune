import torch
from transformers import AutoModelForCausalLM, GPT2Tokenizer

model_name = "gpt2-medium_100"

# Load pre-trained GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


def generate_text(prompt,max_length=500):
    input_ids = tokenizer.encode(prompt,return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape,dtype=torch.long)
    pad_token_id = tokenizer.eos_token_id

    generated = model.generate(
        input_ids,
        max_length = max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        attention_mask = attention_mask,
        pad_token_id = pad_token_id
    )

    return tokenizer.decode(generated[0],skip_special_tokens=True)

response = generate_text("Chicken Butter Masala")

print(response)