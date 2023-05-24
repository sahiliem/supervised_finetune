import os

import torch
from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel

#https://huggingface.co/blog/how-to-generate

text = "A long time ago, in a Galaxy"

#model_path = os.environ["GENERATED_MODEL"]+'4/model'
#model_path = "./mymodel"
#model_path = "distilgpt2"
#model_path = "gpt2"
model_path = "gpt2-large"

#model_path = "gpt2-medium-2"

tokenizer = AutoTokenizer.from_pretrained(model_path)

#model = GPT2LMHeadModel.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)
model = GPT2LMHeadModel.from_pretrained(model_path)

# encode context the generation is conditioned on
#input_ids = tokenizer.encode(text, return_tensors='pt')

# print("================Greedy=============")
# # generate text until the output length (which includes the context length) reaches 50
# greedy_output = model.generate(input_ids, max_length=150)
#
# print("Output:\n" + 100 * '-')
# print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
#
# print("================Beam=============")
# # activate beam search and early_stopping
# beam_outputs = model.generate(
#     input_ids,
#     max_length=150,
#     num_return_sequences=5,
#     num_beams=5,
#     early_stopping=True,
#     no_repeat_ngram_size=2
# )
#
# print("Output:\n" + 100 * '-')
# #print(tokenizer.decode(beam_outputs[0], skip_special_tokens=True))
# for i, beam_output in enumerate(beam_outputs):
#   print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))


#
# print("=======Top K===============")
#
# sample_output = model.generate(
#     input_ids,
#     do_sample=True,
#     max_length=300,
#     top_k=50
# )
#
# print("Output:\n" + 100 * '-')
# print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
#
# print("=======Top_p==============")
#
# sample_output = model.generate(
#     input_ids,
#     do_sample=True,
#     max_length=300,
#     top_p=0.92,
#     top_k=0
# )
#
# print("Output:\n" + 100 * '-')
# print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
#
# print("num_return_sequences top_p and top_k")
# # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
# sample_outputs = model.generate(
#     input_ids,
#     do_sample=True,
#     max_length=300,
#     top_k=50,
#     top_p=0.95,
#     num_return_sequences=3
# )
#
# print("Output:\n" + 100 * '-')
# for i, sample_output in enumerate(sample_outputs):
#   print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
#   print("\n" + 10 * '-')



print("Output:\n" + 100 * '-')

def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    pad_token_id = tokenizer.eos_token_id

    generated = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id
    )

    return tokenizer.decode(generated[0], skip_special_tokens=True)

resp = generate_text(text,300)
print(resp)