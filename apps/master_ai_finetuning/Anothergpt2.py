import os

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments


from master_ai_finetuning.AnotherCustomDataSet import CustomTextDataset

#https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb

# start a new wandb run to track this script
run  = wandb.init(
    # set the wandb project where this run will be logged
    project="AnotherGPTXXXX",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 1e-5,
        "architecture": "gpt2-medium",
        "epochs": 50,
        "max_length":1024,
        "batch_size":8
    }
)

train_path = 'train_desi_dish_dataset.text'
test_path = 'test_desi_dish_dataset.text'

model_name = "gpt2-medium"

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

# Define training arguments
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Set the number of iterations after which to save a checkpoint
checkpoint_interval = 100
iteration = 0

# Training loop
num_epochs = 1
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

        iteration += 1

        # Save a checkpoint every checkpoint_interval iterations
        if iteration % checkpoint_interval == 0:
            checkpoint_path = os.environ["GENERATED_MODEL"] + f"/"+model_name+"/checkpoint_"+{iteration}+".pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at iteration {iteration}")

    # Print average loss for the epoch
    average_loss = total_loss / len(train_data_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss:.4f}")
    # log metrics to wandb
    run.log({'epoch': epoch+1, "train_loss": average_loss})

    # Evaluation
    model.eval()
    total_eval_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in eval_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            total_eval_loss += loss.item()
            total_tokens += torch.numel(input_ids)

    # Print average evaluation loss for the epoch
    average_eval_loss = total_eval_loss / len(eval_data_loader)
    perplexity = torch.exp(torch.tensor(total_eval_loss / total_tokens))
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Evaluation Loss: {average_eval_loss:.4f}")
    print(f"Epoch {epoch + 1}/{num_epochs}, Perplexity: {perplexity:.4f}")
    run.log({'epoch': epoch + 1, "eval_loss": average_loss})
    run.log({'epoch': epoch + 1, "perplexity": perplexity})


#Save the model
torch.save(model.state_dict(), os.environ["GENERATED_MODEL"]+"/"+model_name+"_"+str(num_epochs)+".pth")

model.save_pretrained(os.environ["GENERATED_MODEL"]+"/"+model_name+"_"+str(num_epochs))
tokenizer.save_pretrained(os.environ["GENERATED_MODEL"]+"/"+model_name+"_"+str(num_epochs))


artifact = wandb.Artifact('model', type='model')
artifact.add_file(os.environ["GENERATED_MODEL"]+"/"+model_name+"_"+str(num_epochs)+".pth")
run.log_artifact(artifact)

run.finish()