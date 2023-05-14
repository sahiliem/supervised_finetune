import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split

import re

# Create a regular expression to match wrong encoded text
regex = re.compile(r'[^\x00-\x7F]')

def matched_encoded_text(text):
    matches = regex.findall(text)

    # Print the list of invalid characters
    if matches:
        return True
    else:
        return False


dataframe = pd.read_csv("IndianFoodDatasetCSV.csv")
print(dataframe.keys())


# with open('recipes.json') as f:
#     data = json.load(f)

def build_text_files(data_json, dest_path):
    data = []
    for text in data_json:
        text = str(text).strip()
        if matched_encoded_text(text) is False:
            summary = text
            summary = re.sub(r"\s", " ", summary)
            data.append(summary)

    my_dict = {"texts":data}
    # Save the dictionary to a JSON file
    with open(dest_path, "w",encoding="utf-8") as f:
        json.dump(my_dict, f, indent=4)
        f.close()

train, test = train_test_split(dataframe['TranslatedInstructions'],test_size=0.15)


build_text_files(train,'train_indian_dataset.json')
build_text_files(test,'test_indian_dataset.json')

print("Train dataset length: "+str(len(train)))
print("Test dataset length: "+ str(len(test)))