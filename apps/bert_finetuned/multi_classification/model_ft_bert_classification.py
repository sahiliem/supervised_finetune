import os

import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

from apps.bert_finetuned.multi_classification.train_test_data import train_dataset, val_dataset, label_map
from apps.initialize import init

init()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased",num_labels=4)

def tokenize_function(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# # Convert the validation dataset to Tensors
# val_inputs = [tokenizer(val[0], padding="max_length", truncation=True, max_length=128, return_tensors="pt") for val in val_dataset]
# val_labels = [torch.tensor(label_map[val[1]]) for val in val_dataset]
# val_dataset = list(zip(val_inputs, val_labels))

# Define the training and validation datasets
# train_dataset = [(input_dict, label) for input_dict, label in your_train_data]
# val_dataset = [(input_dict, label) for input_dict, label in your_val_data]

# Define the data collator function
def data_collator(data):
    input_ids = [tokenize_function(item[0])['input_ids'] for item in data]
    attention_mask = [tokenize_function(item[0])['attention_mask'] for item in data]
    labels = [label_map[item[1]] for item in data]

    model_inputs = torch.stack(input_ids)
    model_inputs = model_inputs.view(-1, model_inputs.size(-1))
    model_attention_mask = torch.stack(attention_mask)
    model_attention_mask = model_attention_mask.view(-1, model_attention_mask.size(-1))

    return {'input_ids': model_inputs, 'attention_mask': model_attention_mask, 'labels': torch.tensor(labels)}

# Define the trainer and training arguments
training_args = TrainingArguments(
    output_dir=os.environ["GENERATED_MODEL"]+'/results',
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=100,
    weight_decay=0.01,
    report_to="wandb"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model
model.save_pretrained(os.environ["GENERATED_MODEL"]+'/model')
tokenizer.save_pretrained(os.environ["GENERATED_MODEL"]+'/model')

wandb.finish()