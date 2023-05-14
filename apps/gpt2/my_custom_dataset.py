import json
import os
import torch

from transformers import GPT2Tokenizer


class CustomTextDataset(torch.utils.data.Dataset):
    """
    Custom text dataset for text generation.

    Args:
        tokenizer (GPT2Tokenizer): The GPT-2 tokenizer.
        file_path (str): The directory containing the text data.
        block_size (int): The maximum length of a sequence of tokens that will be processed by the model at once.
    """

    def __init__(self, tokenizer:GPT2Tokenizer, file_path:str, block_size:int):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        self.texts = []
        # for filename in os.listdir(data_dir):
        #     with open(os.path.join(data_dir, filename), "r") as f:
        #         text = f.read()
        #         self.texts.append(text)
        with open(file_path, "r",encoding="utf-8") as f:
            my_data = json.load(f)
            self.texts.extend(my_data["texts"])
            length = len(self.texts)
            print(length)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        self.block_size = self.block_size - self.tokenizer.num_special_tokens_to_add(pair=False)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        encoded_inputs = self.tokenizer(text=text,
                                        max_length=self.block_size,
                                        truncation=True,
                                        padding="max_length",
                                        add_special_tokens=True)
        return torch.tensor(encoded_inputs["input_ids"], dtype=torch.long)

