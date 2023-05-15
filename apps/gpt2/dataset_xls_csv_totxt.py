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

def ingredients(x:str):
    return str(x) + ".\n "
def instructions(x:str):
    return "The instructions recipes follows as : \n"+str(x)

def recipeName(x:str):
    return "The ingredients for the recipe " + str(x) + " are : \n"

s0 = dataframe['TranslatedRecipeName'].apply(recipeName)
s1 = dataframe['TranslatedIngredients'].apply(ingredients)
s2 = dataframe['TranslatedInstructions'].apply(instructions)
dd  = s0 + s1 + s2
print(dd)


# with open('recipes.json') as f:
#     data = json.load(f)

def build_text_files(data_json, dest_path):
    f = open(dest_path, 'w', encoding="utf-8")
    data = ''
    for text in data_json:
        text = str(text).strip()
        if matched_encoded_text(text) is False:
            summary = text
            summary = re.sub(r"\s", " ", summary)
            data += summary + "  "
    f.write(data)

train, test = train_test_split(dd,test_size=0.15)


build_text_files(train,'train_desi_dish_dataset.text')
build_text_files(test,'test_desi_dish_dataset.text')

print("Train dataset length: "+str(len(train)))
print("Test dataset length: "+ str(len(test)))