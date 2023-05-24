import torch
import wandb
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from initialize import init

init()


train_path = 'train_desi_dish_dataset.text'
test_path = 'test_desi_dish_dataset.text'

# Load the GPT-2 medium model
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

# Set the learning rate
learning_rate = 0.0001

# Set the batch size
batch_size = 16

# Set the warm-up steps
warmup_steps = 1000

# Set the decay steps
decay_steps = 10000

# Set the loss function
loss_function = torch.nn.CrossEntropyLoss()

# Set the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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

# Set the training data
training_data = train_dataset

# Set the validation data
validation_data = eval_dataset

# Train the model
for epoch in range(10):

    # Train the model on a batch of data
    model.train()
    for batch in training_data:
        inputs = batch["input_ids"]
        labels = batch["labels"]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluate the model on a batch of data
    model.eval()
    for batch in validation_data:
        inputs = batch["input_ids"]
        labels = batch["labels"]
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

    # Print the loss
    print(loss)

# Save the model
model.save_pretrained("gpt2-medium-finetuned")
