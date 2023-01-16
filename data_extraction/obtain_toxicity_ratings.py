from os import getpid
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
from math import ceil
import psutil
from torch import sigmoid

tokenizer = BertTokenizer.from_pretrained("unitary/toxic-bert")
text_model = BertForSequenceClassification.from_pretrained("unitary/toxic-bert", output_hidden_states=True).to(device="cuda")

dataset_name = "big_dataset_nov22"

df = pd.read_csv(f"data_extraction/{dataset_name}.csv_informative")
print(len(df))
print(df.nunique())

def embed_texts(batch):
    outputs = text_model(tokenizer(batch,padding=True,truncation=True, return_tensors="pt")["input_ids"].to(device="cuda"))
    batch_scores = sigmoid(outputs.logits).cpu().detach().numpy()[:,0]
    batch_embeds = np.average(outputs["hidden_states"][-1].cpu().detach().numpy(),axis=1)
    
    print(outputs["hidden_states"][-1].cpu().detach().numpy().shape)
    input()
    return batch_scores, batch_embeds
batch_size=4

comments=np.array_split(df['Body'],ceil(len(df)/batch_size))
print(df)
input()
embeddings = np.zeros((len(df), 768))
scores = np.zeros(len(df))

running_total=0

for i, batch in enumerate(comments):

        mem = psutil.Process(getpid()).memory_info().rss / 1024 ** 2

        if i%10==0:
            print(f"\033[K Processing texts... Batch {i + 1}/{len(comments)} | Batch size {len(batch)} | Pos {i*batch_size}..{i*batch_size+len(batch)} , Memory usage: {mem} MB",
                end="\r")

        batch_scores, batch_embeds = embed_texts(list(batch))
        embeddings[running_total:running_total+len(batch)] = batch_embeds
        scores[running_total:running_total+len(batch)] = batch_scores
        running_total+=len(batch)


