import numpy as np
import pandas as pd
from text_utils import preprocess_text
import tqdm 
import pickle5

#Load preprocessed dataset. Contains the "informative" comments, 
#but they are not text-preprocessed (lemmatization, stopword removal etc)
df=pd.read_csv('preprocessed_datasets/coronavirus_2021q1_all_preprocessed.csv',encoding='UTF_8')
print(df)

#Obtain the preprocessed texts
tqdm.tqdm.pandas()
df['preprocessed_body'] = df['Body'].progress_apply(preprocess_text)

df.drop(['Body'], axis=1).to_csv(f'preprocessed_datasets/coronavirus_2021q1_all_preprocessed_texts.csv', encoding='utf-8', index=False)

print(df['Body'].head(5))
print(df['preprocessed_body'].head(5))

#Obtain the global dictionary
global_dict={}

filter_words=["//","r/",".com",".org"]

for i,row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    for word in row["preprocessed_body"]:
        if not any([w in word for w in filter_words]):
            if word not in global_dict:
                global_dict[word] = {(row['author_id'],row['subreddit_id'])}
            else:
                global_dict[word].add((row['author_id'],row['subreddit_id']))
global_dict = dict(sorted(global_dict.items(), key=lambda item: len(item[1]),reverse=True))

print(len(global_dict))     #315k unique terms
print(str(global_dict)[:1000])


#We remove from the global dictionary all words that appear in <20 users and 
#<5 subreddits. This helps reduce dimensionality
keys=list(global_dict)
for word in tqdm.tqdm(keys):
    if len(set([author_id for (author_id,_) in global_dict[word]]))<20 or len(set([subreddit_id for (_,subreddit_id) in global_dict[word]]))<5:
        global_dict.pop(word)
    
print(len(global_dict))     #11k unique terms
print(str(global_dict)[:1000])

#Create array for storing the BOW of each user and each subreddit
user_row_bows = np.zeros((df['author_id'].unique().shape[0],len(global_dict)))
subreddit_row_bows = np.zeros((df['subreddit_id'].unique().shape[0],len(global_dict)))

#Store the appeareances of each keyword in the user's or subreddit's BOW
for i,(key,values) in enumerate(global_dict.items()):
    for (subreddit_id,author_id) in values:
        user_row_bows[row['author_id'],i]+=1
        subreddit_row_bows[row['subreddit_id'],i]+=1

#Save user BOWs and subreddit BOWs
with open('preprocessed_datasets/coronavirus_2021q1_all_preprocessed_USERS_BAG_OF_WORDS', 'wb') as handle:
    pickle5.dump(user_row_bows, handle, protocol=pickle5.HIGHEST_PROTOCOL)

with open('preprocessed_datasets/coronavirus_2021q1_all_preprocessed_SUBREDDIT_BAG_OF_WORDS', 'wb') as handle:
    pickle5.dump(subreddit_row_bows, handle, protocol=pickle5.HIGHEST_PROTOCOL)