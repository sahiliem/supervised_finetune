import os

import wandb
from transformers import AutoModelForCausalLM, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from initialize import init

#https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb

init()

train_path = 'train_desi_dish_dataset.text'
test_path = 'test_desi_dish_dataset.text'

# Load pre-trained GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained('distilgpt2')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# Load training data as TextDataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path, # path to training data file
    block_size=128 # maximum sequence length
)

# Load validation data as TextDataset
eval_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=test_path, # path to validation data file
    block_size=128 # maximum sequence length
)

# Define data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=os.environ["GENERATED_MODEL"] + '/results', # output directory
    overwrite_output_dir=True,
    num_train_epochs=1, # number of training epochs
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='steps',
    eval_steps=500, # evaluate every 500 steps
    save_total_limit=2,
    save_steps=500, # save every 500 steps
    warmup_steps=500,
    learning_rate=2e-5,
    #fp16=True, # use mixed precision training
    logging_dir=os.environ["GENERATED_MODEL"] +'/logs',
    logging_steps=500,
    report_to="wandb"
)

# Create Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    eval_dataset=eval_dataset,
)

# Fine-tune the model
trainer.train()

# Define a function to compute perplexity during evaluation
def compute_perplexity(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    loss = trainer.compute_loss(eval_pred)
    perplexity = loss.exp()
    return {'perplexity': perplexity}

# Set the compute_metrics argument in Trainer to use the perplexity function
trainer.compute_metrics = compute_perplexity

# Evaluate the model and calculate perplexity
evaluation_result = trainer.evaluate()

print("Perplexity:", evaluation_result["perplexity"])


#Save the model
model.save_pretrained(os.environ["GENERATED_MODEL"]+'/model3')
tokenizer.save_pretrained(os.environ["GENERATED_MODEL"]+'/model3')

wandb.finish()