import os

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments


from initialize import init
from master_ai_finetuning.AnotherCustomDataSet import CustomTextDataset

#https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-pytorch-project",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 1e-5,
        "architecture": "distilgpt2",
        "epochs": 10,
        "max_length":1024,
        "batch_size":8
    }
)

train_path = 'train_desi_dish_dataset.text'
test_path = 'test_desi_dish_dataset.text'

model_name = "distilgpt2"
model_name = "distilgpt2_1"

# Load pre-trained GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset = CustomTextDataset(train_path, tokenizer, max_length=1024)
eval_dataset = CustomTextDataset(test_path, tokenizer, max_length=1024)

# Create data loader
batch_size = 8
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)


# # Load training data as TextDataset
# train_dataset = TextDataset(
#     tokenizer=tokenizer,
#     file_path=train_path, # path to training data file
#     block_size=128 # maximum sequence length
# )
#
# # Load validation data as TextDataset
# eval_dataset = TextDataset(
#     tokenizer=tokenizer,
#     file_path=test_path, # path to validation data file
#     block_size=128 # maximum sequence length
# )
#
# # Define data collator for language modeling
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer, mlm=False
# )

# Define training arguments
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print average loss for the epoch
    average_loss = total_loss / len(train_data_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")
    # log metrics to wandb
    wandb.log({'epoch': epoch+1, "train_loss": average_loss})

    # Evaluation
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in eval_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            total_eval_loss += loss.item()

    # Print average evaluation loss for the epoch
    average_eval_loss = total_eval_loss / len(eval_data_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Evaluation Loss: {average_eval_loss:.4f}")
    wandb.log({'epoch': epoch + 1, "eval_loss": average_loss})

# Define a function to compute perplexity during evaluation
# def compute_perplexity(eval_pred):
#     logits, labels = eval_pred.predictions, eval_pred.label_ids
#     loss = trainer.compute_loss(eval_pred)
#     perplexity = loss.exp()
#     return {'perplexity': perplexity}
#
# # Set the compute_metrics argument in Trainer to use the perplexity function
# trainer.compute_metrics = compute_perplexity
#
# # Evaluate the model and calculate perplexity
# evaluation_result = trainer.evaluate()
#
# print("Perplexity:", evaluation_result["perplexity"])


#Save the model
model.save_pretrained(os.environ["GENERATED_MODEL"]+'/../../distilgpt2_2')
tokenizer.save_pretrained(os.environ["GENERATED_MODEL"]+'/../../distilgpt2_2')

wandb.finish()