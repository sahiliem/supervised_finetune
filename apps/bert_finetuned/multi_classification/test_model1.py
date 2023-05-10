import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

text = "The shipping address is B307, Kudlu"

tokenizer = AutoTokenizer.from_pretrained('model')
model = AutoModelForSequenceClassification.from_pretrained('model')

encoded_input = tokenizer(text, return_tensors='pt')
#labels = torch.tensor([1]).unsqueeze(0)
output = model(**encoded_input)
print(output)


pipe = pipeline("text-classification",model ='model',tokenizer=tokenizer)

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