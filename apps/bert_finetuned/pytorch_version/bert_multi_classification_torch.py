# Import required libraries
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, random_split, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

# Load dataset
df = pd.read_csv('dataset.csv')
sentences = df['text'].values
labels = df['label'].values

# Tokenize input sentences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
input_ids = []
attention_masks = []
for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad or truncate all sentences
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks
                        return_tensors = 'pt',      # Return pytorch tensors
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# Convert input_ids, attention_masks, and labels to tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Create TensorDataset
dataset = TensorDataset(input_ids, attention_masks, labels)

# Split dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoader for each set
batch_size = 32
train_dataloader = DataLoader(train_dataset, sampler=None, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=None, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=None, batch_size=batch_size)

# Load pre-trained BERT model and set number of output classes
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Set device to run on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Train model
def train_model():
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)

            model.zero_grad()
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)

        # Validate model
        model.eval()
        total_val_loss = 0
        for batch in val_dataloader:
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
                loss = outputs.loss
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)

            # Print progress
            print(f'Epoch {epoch + 1}:')
            print(f'Training Loss: {avg_train_loss:.3f}')
            print(f'Validation Loss: {avg_val_loss:.3f}')

        # Test model
        model.eval()

        total_test_loss = 0
        predictions, true_labels = [], []
        for batch in test_dataloader:
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
                loss = outputs.loss
                logits = outputs.logits
                total_test_loss += loss.item()

            # Convert logits to predictions
            logits = logits.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()
            predictions.extend(logits)
            true_labels.extend(label_ids)

        avg_test_loss = total_test_loss / len(test_dataloader)

        # Calculate accuracy
        from sklearn.metrics import accuracy_score

        pred_labels = [np.argmax(p) for p in predictions]
        acc = accuracy_score(true_labels, pred_labels)
        print(f'Test Loss: {avg_test_loss:.3f}')
        print(f'Test Accuracy: {acc:.3f}')



# Save model state dict and optimizer state dict
torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'bert_model.pth')



# Load saved model state dict and optimizer state dict
checkpoint = torch.load('bert_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

