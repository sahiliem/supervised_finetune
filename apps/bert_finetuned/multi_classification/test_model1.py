import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_path = os.environ["GENERATED_MODEL"]+'/model'

text = "invoiceId amount Item_price"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

encoded_input = tokenizer(text, return_tensors='pt')
#labels = torch.tensor([1]).unsqueeze(0)
output = model(**encoded_input)
print(output)


pipe = pipeline("text-classification",model =model_path,tokenizer=tokenizer)

output_pipe = pipe(text)
print(output_pipe)

# Tokenize input text
inputs = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    return_tensors='pt',
)

# Make prediction
outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
predicted_label = torch.argmax(probs, dim=-1).item()

# Print predicted label
print('Predicted label:', predicted_label)