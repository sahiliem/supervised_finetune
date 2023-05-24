import os

import torch
import wandb
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

mode_name = "distilgpt2"
# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained(mode_name)
tokenizer = GPT2Tokenizer.from_pretrained(mode_name)
tokenizer.pad_token = tokenizer.eos_token

train_path = 'train_desi_dish_dataset.text'
test_path = 'test_desi_dish_dataset.text'

from datasets import load_dataset

#dataset = load_dataset("recipe_nlg",data_dir=os.environ["DATASET_STORAGE"])
dataset = load_dataset("wikitext",'wikitext-103-v1')

tokenized_dataset = dataset.map(lambda e:tokenizer(e['text'],truncation=True,padding="max_length"))


print(model.config)

def generate_text(prompt,max_length=50):
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