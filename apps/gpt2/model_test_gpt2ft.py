import os

from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel

#https://huggingface.co/blog/how-to-generate

text = "How to make Punjabi lobiya masala?"

model_path = os.environ["GENERATED_MODEL"]+'/model'
model_path = "./mymodel"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = GPT2LMHeadModel.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)

# encode context the generation is conditioned on
input_ids = tokenizer.encode(text, return_tensors='pt')

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



print("=======Top K===============")

sample_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=150,
    top_k=50
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

print("=======Top_p==============")

sample_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=150,
    top_p=0.92,
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

print("num_return_sequences top_p and top_k")
# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
sample_outputs = model.generate(
    input_ids,
    do_sample=True,
    max_length=150,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))