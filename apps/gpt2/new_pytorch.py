import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained GPT-2 medium model
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.to(device)

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

# Define hyperparameters
epochs = 5
batch_size = 8
learning_rate = 2e-5
warmup_steps = 1000
total_steps = 5000
gradient_accumulation_steps = 4
decay_factor = 0.9

# Define optimizer and scheduler
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     "weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# Fine-tuning loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    total_steps = 0
    for step, batch in enumerate(train_dataloader):
        input_ids = batch.to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

        # Calculate gradients
        loss.backward()

        # Update parameters
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_steps += 1

    average_loss = total_loss / total_steps
    print(f"Epoch: {epoch + 1} - Average Loss: {average_loss}")

    # Early stopping condition (monitor loss and accuracy)
    if average_loss < 0.001:
        print("Training stopped due to convergence.")
        break

    # Adjust learning rate using decay schedule
    learning_rate *= decay_factor
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
