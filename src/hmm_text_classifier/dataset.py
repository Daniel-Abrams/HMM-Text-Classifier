import pandas as pd
from pathlib import Path
from cleantext import clean
import ast
import json
import re
from bpetokenizer import BPETokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def get_dataset():
    splits = {'train': 'data/mixed/train.parquet', 'validation': 'data/mixed/validation.parquet', 'test': 'data/mixed/test.parquet'}
    train_df = pd.read_parquet("hf://datasets/open-text-detector/heterogeneous-ai-spans/" + splits["train"])
    validation_df = pd.read_parquet("hf://datasets/open-text-detector/heterogeneous-ai-spans/" + splits["validation"])
    test_df = pd.read_parquet("hf://datasets/open-text-detector/heterogeneous-ai-spans/" + splits["test"])

    train_df.to_csv(PROJECT_ROOT / "data" / "train_dataset.csv", index=False)
    validation_df.to_csv(PROJECT_ROOT / "data" / "validation_dataset.csv", index=False)
    test_df.to_csv(PROJECT_ROOT / "data" / "test_dataset.csv", index=False)



def trainTokenizer():
    df = pd.read_csv(PROJECT_ROOT / "data" / "train_dataset.csv")
    tokenizer = BPETokenizer() 
    
    df = df[df['split'] == 'train']
    res = []
    full_text = ""
    for item in df.itertuples():
        full_text += " " + item.text
    
    
    tokenizer.train(full_text, vocab_size=3000)
    tokenizer.save("bpetokenizer", mode="json")
    
def tokenizeText(split='train'):
    df = pd.read_csv(PROJECT_ROOT / "data" / f"{split}_dataset.csv")
    tokenizer = BPETokenizer() 
    tokenizer.load("bpetokenizer.json", mode="json") 

    res = []
    for item in df.itertuples():
        text = item.text
        
        # Need to add in the commas
        spans_string = re.sub(r"}\s*{", "}, {", item.spans)
        spans = ast.literal_eval(spans_string)
        emission_list = []
        state_list = []
        ## Combine the entire span into a list of words, each with a label
        for span in spans:
            section = text[int(span['start']):int(span['end'])]
            label = span['label']
            tokens = tokenizer.encode(section)
            for token in tokens:
                emission_list.append(token)
                state_list.append(label)
        res.append({
            'id' : item.id,
            'emission_list' : emission_list,
            'state_list' : state_list
        })
        
    with open(PROJECT_ROOT / "data" / f"{split}.json", 'w') as file:
        json.dump(res, file, indent=4)
    
def tokenize_by_words(split='train'):
    df = pd.read_csv(PROJECT_ROOT / "data" / f"{split}_dataset.csv")
    res = []
    for item in df.itertuples():
        text = item.text
        
        # Need to add in the commas
        spans_string = re.sub(r"}\s*{", "}, {", item.spans)
        spans = ast.literal_eval(spans_string)
        emission_list = []
        state_list = []
        ## Combine the entire span into a list of words, each with a label
        for span in spans:
            section = text[int(span['start']):int(span['end'])]
            label = span['label']
            cleaned_section = clean_text(section).split()
            for word in cleaned_section:
                emission_list.append(word)
                state_list.append(label)
        res.append({
            'id' : item.id,
            'emission_list' : emission_list,
            'state_list' : state_list
        })
        
    with open(PROJECT_ROOT / "data" / f"{split}.json", 'w') as file:
        json.dump(res, file, indent=4)

def clean_text(text: str):
    return clean(
    text,
    fix_unicode=True,               # Fixes broken unicode encoding
    to_ascii=True,                  # Converts emojis/symbols to ASCII equivalents
    lower=True,                     # Converts text to lowercase
    no_urls=True,                   # Strips URLs
    no_emails=True,                 # Strips email addresses
    no_punct=True                   # Removes leftover punctuation
)

trainTokenizer()