import datetime
import math
import os
import pickle
from os import getpid, makedirs
from random import choice

import numpy as np
import pandas as pd
import psutil
import tqdm

MODE=None
CORONAVIRUS_FILENAME = "coronavirus_2021q1_all.csv"
OTHER_SUBREDDITS_FILENAME = "big_dataset_nov22.csv"
DATETIME_MIN = datetime.datetime.timestamp(datetime.datetime(2021, 3, 1, 0))
DATASET_NAME = "MARCH_21"
LOAD_PREPROCESSED_DATASET=False
DO_TEXT_PREPROCESSING=True
BERT_MODEL="toxic" #original or toxic

############# STEP 1 : COMMENT SELECTION #############

#If we have a preprocessed dataset already, we can use that (only selects comments that are not generic)
if LOAD_PREPROCESSED_DATASET:
    df = pd.read_csv(f'preprocessed_datasets/{CORONAVIRUS_FILENAME[:-4]}_preprocessed_preprocessed_toxicity.csv')
    print(f"Loaded preprocessed dataset with {df.shape[0]} comments")

else: 
    #Load r/covid dataset, select posts in march 2021
    df_covid_r = pd.read_csv('data_extraction/'+CORONAVIRUS_FILENAME)
    df_covid_r["Publish Date"] = df_covid_r["Publish Date"].apply(
        lambda x: datetime.datetime.timestamp(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")))
    df_covid_r = df_covid_r[df_covid_r["Publish Date"] > DATETIME_MIN]

    #Load dataset with user comments on other subreddits
    df_other_r = pd.read_csv('data_extraction/'+OTHER_SUBREDDITS_FILENAME)
    df_other_r = df_other_r.drop_duplicates(['Post ID'], keep='first')

    print(f"Loaded {CORONAVIRUS_FILENAME} ({df_covid_r.shape[0]} r/covid comments)")
    print(f"Loaded {OTHER_SUBREDDITS_FILENAME} ({df_other_r.shape[0]} comments from other subreddits)")

    print(f"Obtained dataset with {df_other_r.shape[0]} comments.")
    print(f"Unique comments: {df_other_r['Body'].nunique()}")

    #This tries to get rid of auto-moderation comments, e.g. "your comment has been removed due to (...)"
    #This is a trivial approach, eventually need to do it in a way that measures if a user has a lot of almost identical comments
    value_counts = df_covid_r["Author"].value_counts()
    df_covid_r = df_covid_r[~df_covid_r["Author"].isin(value_counts[value_counts >= 2000].index)]
    value_counts = df_other_r["Author"].value_counts()
    df_other_r = df_other_r[~df_other_r["Author"].isin(value_counts[value_counts >= 2000].index)]

    print(f"After filtering auto-moderation comments: {df_covid_r.shape[0]} (r/covid), {df_other_r.shape[0]} (other)")

    ### Tag all comments as "covid-related" or "non-covid-related"
    COVID_KEYWORDS = ["covid", "corona", "sars-cov-2", "vaccin", "antivax", "anti-vax", "vaxx", "pfizer", "moderna", "biontech",
                "jansenn", "sinopharm", "astrazeneca", "sputnik v", "novavax", "flattening the curve", "flatten the curve",
                "immunity", "pandemic", "world health organization", "face mask"]

    def is_covid_related(row):
        row_lowercase = str(row["Body"]).lower()
        if any(k in row_lowercase for k in COVID_KEYWORDS):
            return True
        else:
            return False

    print("Tagging comments as covid related or not")
    df_other_r["is_covid_related"] = df_other_r.apply(lambda row: is_covid_related(row), axis=1)
    df_covid_r["is_covid_related"] = True

    import matplotlib.pyplot as plt
    unique_interactions = df_other_r.drop_duplicates(['author_id','subreddit_id'], keep = 'first')
    authors_per_subreddit = unique_interactions.groupby('subreddit_id').size()
    authors_per_subreddit[authors_per_subreddit>20]=20

    ax = authors_per_subreddit.plot.hist(bins=10,xticks=np.arange(0,20,2))
    ax.set_xlabel('Num. of visiting users')
    ax.set_ylabel('Num. of subreddits')
    plt.tight_layout()
    plt.show()

    input()
    
    ### Select only those comments on subreddits where there's more than 50 unique users talking about covid ###
    print("Selecting comments on subreddits with more than 20 users talking about covid... ", end="")
    # covid_comments = df_other_r[df_other_r["is_covid_related"] == True] 
    # covid_users_per_subreddit = covid_comments.groupby(["Subreddit", "Author"]).size().reset_index().groupby("Subreddit").size().reset_index().rename(columns={0: 'count'})
    # subreddits_with_covid_20_covid_users = covid_users_per_subreddit[covid_users_per_subreddit['count'] >= 20]["Subreddit"]
    # df_other_r = df_other_r[df_other_r["Subreddit"].isin(subreddits_with_covid_20_covid_users)]
    print(f"Before:{df_other_r['Subreddit'].nunique()} subreddits")
    df_other_r = df_other_r.groupby('Subreddit').filter(lambda s: s['Author'].nunique()>=20)
    print(f"After:{df_other_r['Subreddit'].nunique()} subreddits")
    print(f"After:{df_other_r['Author'].nunique()} authors")
    ### Merge r/covid and other subreddis' comments datasets ###
    print("Merging both datasets...")
    df = pd.concat([df_other_r[["Subreddit", "Body", "Author","is_covid_related"]], df_covid_r[["Subreddit", "Body", "Author","is_covid_related"]]]).reset_index(drop=True)

    print(f"Dropping NAs... {df.shape[0]} -> ",end="")
    df = df[df['Body'].notna()]
    print(df.shape[0])

    # print(f"Before:{df_other_r['Author'].nunique()} subreddits")
    # df_other_r = df_other_r.groupby('Author').filter(lambda s: s['Subreddit'].nunique()>5)
    # print(f"After:{df_other_r['Author'].nunique()} subreddits")

    print(f"Obtained dataset with {df.shape[0]} comments.")
    print(f"Unique comments: {df['Body'].nunique()}")


    df = df.assign(author_id=(df["Author"]).astype('category').cat.codes)
    df = df.assign(subreddit_id=(df["Subreddit"]).astype('category').cat.codes)
    df = df.drop(['Subreddit', 'Author'], axis=1)
    df["comment_id"] = np.arange(0, df.shape[0]).tolist()

    # print(df)

    df.to_csv(f'data_extraction/{CORONAVIRUS_FILENAME}_RAWFINAL', encoding='utf-8', index=False)

    input()
    ##########################################################
    #       STEP 2 : PREPARE FOR SAMPLING AND EMBEDDING      #
    ##########################################################

    print("Pickling with protocol ", pickle.HIGHEST_PROTOCOL)
    # makedirs(f"{DATASET_NAME}/IMGMODEL/data_10+10/", exist_ok=True)
    # makedirs(f"{DATASET_NAME}/IMGMODEL/original_take/", exist_ok=True)

    if DO_TEXT_PREPROCESSING:
        from text_utils import preprocess_text, subreddit_keywords

        print("Preprocessing texts...")
        tqdm.tqdm.pandas()
        df['preprocessed_body'] = df['Body'].progress_apply(preprocess_text)

        df['is_informative_comment']=True


        for i,subreddit_id in enumerate(pd.unique(df['subreddit_id'])):
            print(f"Filtering generic comments for subreddit {i}/{df['subreddit_id'].nunique()}...", end='\r')
            specific_subreddit_keywords=subreddit_keywords(df,subreddit_id)
            # A comment is marked as informative if: 
            # a) The comment has at least one subreddit-specific keyword
            # b) The comment talks about Covid
            df["is_informative_comment"]=df.apply(lambda row: row['is_informative_comment'] if subreddit_id!=row['subreddit_id']  else (not set(row['preprocessed_body']).isdisjoint(specific_subreddit_keywords) or subreddit_id!=row['subreddit_id'] or row["is_covid_related"]),axis=1)

        df=df[df["is_informative_comment"]==True]
        
    print(f'Obtained {len(df)} informative comments')
    #Fabricate new author and subreddit id's
    df = df.groupby('subreddit_id').filter(lambda s: s['author_id'].nunique()>=20)
    df = df.groupby('author_id').filter(lambda s: s['subreddit_id'].nunique()>=5)
    df = df.assign(author_id=(df["author_id"]).astype('category').cat.codes)
    df = df.assign(subreddit_id=(df["subreddit_id"]).astype('category').cat.codes)

    print(f'Obtained {len(df)} comments after filtering active subreddits and users')
    df["comment_id"] = np.arange(0, df.shape[0]).tolist()
    
    df = df.drop(['is_informative_comment'],axis=1)

    df.drop(['Body'],axis=1).to_csv(f'data_extraction/{OTHER_SUBREDDITS_FILENAME}_preprocessed_texts', encoding='utf-8', index=False)

    if DO_TEXT_PREPROCESSING:
        print(f"Total unique users: {df['author_id'].unique().shape[0]}")
        print(f"Total unique subreddits: {df['subreddit_id'].unique().shape[0]}")
        print(f"Total samples: {len(df)}")
        print(f"Total interactions: {df.groupby(['author_id','subreddit_it']).ngroups}")

    if DO_TEXT_PREPROCESSING:
        df = df.drop(['preprocessed_body'], axis=1)
        makedirs('preprocessed_datasets',exist_ok=True)
        df.to_csv(f'data_extraction/{OTHER_SUBREDDITS_FILENAME}_informative', encoding='utf-8', index=False)

comments = df["Body"].to_numpy()
df = df.drop(['Body'], axis=1)

##########################################################
#            STEP 3.a) : GENERATE EMBEDDINGS             #
##########################################################
if MODE == "embeds":
    print("Pickling with protocol ", pickle.HIGHEST_PROTOCOL)
    print("Creating ", comments.shape[0], " embeds")

    import gc

    
    
    os.makedirs('BERT_EMBEDDINGS',exist_ok=True)
    # Initialize BERT model
    
    if BERT_MODEL=="original":
        from transformers import TFBertModel, TFBertTokenizer
        text_model = TFBertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        tokenizer = TFBertTokenizer.from_pretrained('bert-base-uncased') 
    elif BERT_MODEL=="toxic":
        print("Generating embeds with toxic-bert model")
        from transformers import BertModel, BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("unitary/toxic-bert")
        text_model = BertModel.from_pretrained("unitary/toxic-bert", output_hidden_states=True).to(device="cuda")
   

    embeddings = np.zeros((comments.shape[0], 768))


    #Function to embed a single text
    if BERT_MODEL=="original":
        def embed_text(text):
            return np.average(text_model(tokenizer(text, return_tensors='tf', padding=True, truncation=True))[0].numpy(),axis=1)
    elif BERT_MODEL=="toxic":
        def embed_text(text):
            return np.average(text_model(tokenizer(text,padding=True,truncation=True, return_tensors="pt")["input_ids"].to(device="cuda"))["last_hidden_state"].cpu().detach().numpy(),axis=1)
            # return text_model(tokenizer(text,padding=True,truncation=True, return_tensors="pt")["input_ids"].to(device="cuda"))["pooler_output"].cpu().detach().numpy()
    #Embed all texts 

    batch_size=8
    comments=np.array_split(comments,math.ceil(comments.shape[0]/batch_size))
    
    running_total=0
    for i, batch in enumerate(comments):

        mem = psutil.Process(getpid()).memory_info().rss / 1024 ** 2

        if i%10==0:
            print(f"\033[K Processing texts... Batch {i + 1}/{len(comments)} | Batch size {len(batch)} | Pos {i*batch_size}..{i*batch_size+len(batch)} , Memory usage: {mem} MB",
                end="\r")

        embeddings[running_total:running_total+len(batch)] = embed_text(list(batch))
        running_total+=len(batch)

    print("\n")
    print("Freeing up memory... Memory Usage:", psutil.Process(getpid()).memory_info().rss / 1024 ** 2)

    # Collect garbage
    del df
    del comments
    texts_mask = None
    images_mask = None
    images = None
    texts = None
    df = None
    comments= None
    gc.collect()

    print("Saving pickle... Memory Usage:", psutil.Process(getpid()).memory_info().rss / 1024 ** 2)
    with open(f"BERT_EMBEDDINGS/{DATASET_NAME}_{BERT_MODEL}_CLS", 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved pickle.")


##########################################################
#               STEP 3.b) : OVERSAMPLING                 #
##########################################################
elif MODE == "sampling": 
    print("Pickling with protocol ", pickle.HIGHEST_PROTOCOL)
    def separate_train_test(data):
        combs = data.groupby(["subreddit_id", "author_id"]).size().reset_index()
        subreddits_per_user = combs.groupby("author_id").size().reset_index().rename(columns={0: 'count'})
        user_with_2_subreddits = subreddits_per_user[subreddits_per_user['count'] >= 2]["author_id"]
        print(f"Splitting {data.shape[0]} rows. Eligible for test: {user_with_2_subreddits.shape[0]} users of {subreddits_per_user.shape[0]}")

        test=[]

        user_groups = data.groupby(['author_id'])
        
        for _, group in user_groups:

                # If they have comments on more than one subreddit, add all their comments to the test set those of the subreddit he has commented the least on
                subreddit_groups = group.groupby(["subreddit_id"])
                comments_per_subreddit = subreddit_groups.size().sort_values(ascending=True).reset_index().rename(columns={0: 'count'})
                if subreddit_groups.ngroups > 1:
                    subreddit_id = comments_per_subreddit[comments_per_subreddit["count"]==comments_per_subreddit["count"].iloc[0]].sample(n=1)["subreddit_id"].iloc[0]
                    rows = group[group["subreddit_id"]==subreddit_id].to_dict(orient="records")
                    for r in rows:
                        test.append(r)
        test = pd.DataFrame(test)
        print(f"Total test samples: {test.shape[0]}")

        train = pd.concat([data, test]).drop_duplicates(keep=False)

        return train, test


    train_and_val, test = separate_train_test(df)
    train, val = separate_train_test(train_and_val)

    information = ""
    information += "#################################\n"
    information += str(DATASET_NAME + "\n")
    information += "#################################\n"
    information += "\n"
    information += "====== BEFORE OVERSAMPLING ======\n"
    information += " \t\tREVIEWS\tUSERS\tRESURANTS\n"
    information += str("ALL:\t\t" + str(df.shape[0]) + "\t" + str(pd.unique(df["author_id"]).shape[0]) + "\t" + str(
        pd.unique(df["subreddit_id"]).shape[0]) + "\n")
    information += str("TRAIN:\t\t" + str(train.shape[0]) + "\t" + str(pd.unique(train["author_id"]).shape[0]) + "\t" + str(
        pd.unique(train["subreddit_id"]).shape[0]) + "\n")
    information += str("TRAIN_DEV:\t" + str(train_and_val.shape[0]) + "\t" + str(
        pd.unique(train_and_val["author_id"]).shape[0]) + "\t" + str(
        pd.unique(train_and_val["subreddit_id"]).shape[0]) + "\n")
    information += str("DEV:\t\t" + str(val.shape[0]) + "\t" + str(pd.unique(val["author_id"]).shape[0]) + "\t" + str(
        pd.unique(val["subreddit_id"]).shape[0]) + "\n")
    information += str("TEST:\t\t" + str(test.shape[0]) + "\t" + str(pd.unique(test["author_id"]).shape[0]) + "\t" + str(
        pd.unique(test["subreddit_id"]).shape[0]) + "\n")

    print(information)
    
    train["num_images"] = 1
    train_and_val["num_images"] = 1
    val["num_images"] = 1
    test["num_images"] = 1
    df["num_images"] = 1

    with open(DATASET_NAME + "/IMGMODEL/original_take/TRAIN_DEV", "wb") as f:
        pickle.dump(train_and_val, f)
    with open(DATASET_NAME + "/IMGMODEL/original_take/TRAIN_TEST", "wb") as f:
        pickle.dump(df, f)
    with open(DATASET_NAME + "/IMGMODEL/original_take/DEV", "wb") as f:
        pickle.dump(val, f)
    with open(DATASET_NAME + "/IMGMODEL/original_take/TEST", "wb") as f:
        pickle.dump(test, f)

    # ==================================
    # Mark all existing reviews as positive samples
    # ==================================
    train = train.drop(["num_images"], axis=1)
    train_and_val = train_and_val.drop(["num_images"], axis=1)
    val = val.drop(["num_images"], axis=1)
    test = test.drop(["num_images"], axis=1)

    train["take"] = 1
    train_and_val["take"] = 1
    val["is_dev"] = 1
    test["is_dev"] = 1

    print(test.shape)

    # Sanity check
    merge = train_and_val.merge(test, left_on=["author_id", "subreddit_id", "comment_id"],
                                right_on=["author_id", "subreddit_id", "comment_id"])
    merge1 = train.merge(val, left_on=["author_id", "subreddit_id", "comment_id"],
                         right_on=["author_id", "subreddit_id", "comment_id"])
    print(f"Overlapping samples: {merge.shape[0]} (TRAIN_DEV - TEST), {merge1.shape[0]} (TRAIN - DEV)")


    def oversample_trainset1(train,test=0):

        
        subreddit_groups = train.groupby(["subreddit_id"])
        subreddit_ids=pd.unique(train["subreddit_id"]).tolist()

        i=0
        rows=[]

        for _, row in train.iterrows():
            if i%100==0: print(f"Oversampling train samples... {i}/{train.shape[0]}       ",end="\r")

            row=row.to_dict()
            #For each review (u,r), add then negative samples (u,r') where r' is a photo of the **same** restaurant taken by a different user u'

            sampled_rows=subreddit_groups.get_group((row["subreddit_id"])).copy()
            sampled_rows=sampled_rows[sampled_rows["author_id"]!=row["author_id"]]

            if not sampled_rows.empty: 
                sampled_rows=sampled_rows.sample(n=10,replace=True)
                sampled_rows["author_id"]=row["author_id"]
                sampled_rows["take"]=0
                sampled_rows=sampled_rows.to_dict(orient='records')

                for e in sampled_rows:
                    rows.append(e)
                    rows.append(row)

            #For each review (u,r), add then negative samples (u,r') where r' is a photo of a **different** restaurant taken by a different user u'
            different_subreddit_id=row["subreddit_id"]
            for _ in range(10):
                while different_subreddit_id==row["subreddit_id"]:
                    different_subreddit_id = choice(subreddit_ids)
                sampled_rows=subreddit_groups.get_group(different_subreddit_id).copy()
                sampled_rows=sampled_rows[sampled_rows["author_id"]!=row["author_id"]]
                if not sampled_rows.empty:
                    sampled_rows=sampled_rows.sample(n=1)
                    sampled_rows["author_id"]=row["author_id"]
                    sampled_rows["take"]=0
                    sampled_rows=sampled_rows.to_dict(orient='records')
                    for e in sampled_rows:
                        rows.append(e)    
                        rows.append(row)

            #Note we're adding a replica of the original sample (rows.append(row)) each time we add a negative sample

            i+=1

        print(f"Original: {train.shape[0]} samples | Oversampled: {len(rows)} samples",end="\r")

        return pd.DataFrame.from_records(rows)
    
    def oversample_trainset_old(train):
        rows = []
        newtrain = train.copy()
        i = 0
        for _, row in train.iterrows():
            if i%100==0: print(f"Oversampling train samples... {i}/{train.shape[0]}       ",end="\r")

            row=row.to_dict()

            # For each review (u,r), add then negative samples (u,r') where r' is a photo of the **same** restaurant taken by a different user u'
            same_subreddit = (train.loc[(train["author_id"] != row["author_id"]) & (train["subreddit_id"] == row["subreddit_id"])]).copy()
            if not same_subreddit.empty: same_subreddit = same_subreddit.sample(n=10, replace=True)
            same_subreddit["author_id"] = row["author_id"]
            same_subreddit["take"] = 0
            same_subreddit = same_subreddit.to_dict(orient='records')

            for e in same_subreddit:
                rows.append(e)
                rows.append(row)

            # For each review (u,r), add then negative samples (u,r') where r' is a photo of a **different** restaurant taken by a different user u'
            different_restaurant = (train.loc[
                (train["author_id"] != row["author_id"]) & (train["subreddit_id"] != row["subreddit_id"])]).copy()
            different_restaurant = different_restaurant.sample(n=10, replace=True)
            different_restaurant["author_id"] = row["author_id"]
            different_restaurant["take"] = 0

            different_restaurant = different_restaurant.to_dict(orient='records')
            for e in different_restaurant:
                rows.append(e)
                rows.append(row)
        
            i += 1

        print(f"Original: {train.shape[0]} samples | Oversampled: {len(rows)} samples",end="\r")

        return pd.DataFrame.from_records(rows)

    print("")
    oversampled_train_and_val = oversample_trainset_old(train_and_val)
    with open(DATASET_NAME + "/IMGMODEL/data_10+10/TRAIN_DEV_TXT", "wb") as f:
        pickle.dump(oversampled_train_and_val, f)

    # oversampled_train = oversample_trainset_old(train)
    # with open(DATASET_NAME + "/IMGMODEL/data_10+10/TRAIN_TXT", "wb") as f:
    #     pickle.dump(oversampled_train, f)

    print("")

    def oversample_testset1(test,train):

        subreddit_groups = train.groupby("subreddit_id")
        subreddit_ids=pd.unique(train["subreddit_id"]).tolist()

        i=0
        id_test=0

        rows=[]
        
        for _, row in test.iterrows():
            if i%100==0: print(f"Negative-sampling test samples... {i}/{train.shape[0]}       ",end="\r")

            r=row.to_dict()
            r["id_test"]=id_test

            if (row["subreddit_id"]) in subreddit_ids:

                subreddit=subreddit_groups.get_group(row["subreddit_id"]).copy()
                subreddit=subreddit[subreddit["author_id"]!=row["author_id"]]
                if subreddit.shape[0]>100: subreddit=subreddit.sample(n=100)

                subreddit["author_id"]=row["author_id"]
                subreddit.rename(columns={'take':'is_dev'}, inplace=True)
                subreddit["is_dev"]=0
                subreddit["id_test"]=id_test

                if subreddit.shape[0]!=0:
                    rows.append(r)
                    subreddit=subreddit.to_dict(orient='records')
                    for e in subreddit:
                        rows.append(e)
                    id_test+=1

        print(f"Original: {train.shape[0]} samples | Negative-sampled: {len(rows)} samples",end="\r")
        i+=1

        return pd.DataFrame.from_records(rows)

    def oversample_testset_old(test, train):
        rows = []
        id_test = 0
        for _, row in test.iterrows():
            if id_test%100==0: print(f"Negative-sampling test samples... {id_test}/{train.shape[0]}       ",end="\r")

            id_test += 1
            r = row.to_dict()
            r["id_test"] = id_test

            # For each review (u,r), add negative samples (u,r') for all photos r' taken of the **same** restaurant by a different user.
            same_subreddit = (train.loc[
                (train["author_id"] != row["author_id"]) & (train["subreddit_id"] == row["subreddit_id"])])

            if same_subreddit.shape[0] > 100: same_subreddit = same_subreddit.sample(n=100)

            same_subreddit["author_id"] = row["author_id"]
            same_subreddit.rename(columns={'take': 'is_dev'}, inplace=True)
            same_subreddit["is_dev"] = 0
            same_subreddit["id_test"] = id_test

            if same_subreddit.shape[0] != 0:
                same_subreddit = same_subreddit.to_dict(orient='records')
                rows.append(r)
                for e in same_subreddit:
                    rows.append(e)


        print(f"Original: {train.shape[0]} samples | Negative-sampled: {len(rows)} samples",end="\r")

        return pd.DataFrame.from_records(rows)


    
    # oversampled_val = oversample_testset_old(val, train)
    # with open(DATASET_NAME + "/IMGMODEL/data_10+10/DEV_TXT", "wb") as f:
    #     pickle.dump(oversampled_val, f)

    oversampled_test = oversample_testset_old(test, train_and_val)
    with open(DATASET_NAME + "/IMGMODEL/data_10+10/TEST_TXT", "wb") as f:
        pickle.dump(oversampled_test, f)

    information += "\n"
    information += "====== AFTER OVERSAMPLING ======\n"
    information += "\t\tREVIEWS\tPOSITIVE\tNEGATIVE\n"
    # information += str("TRAIN:\t\t" + str(oversampled_train.shape[0]) + "\t" + str(
    #     oversampled_train.loc[oversampled_train["take"] == 1].shape[0]) + "\t" + str(
    #     oversampled_train.loc[oversampled_train["take"] == 0].shape[0]) + "\n")
    information += str("TRAIN_DEV:\t" + str(oversampled_train_and_val.shape[0]) + "\t" + str(
        oversampled_train_and_val.loc[oversampled_train_and_val["take"] == 1].shape[0]) + "\t" + str(
        oversampled_train_and_val.loc[oversampled_train_and_val["take"] == 0].shape[0]) + "\n")
    # information += str("DEV:\t\t" + str(oversampled_val.shape[0]) + "\t" + str(
    #     oversampled_val.loc[oversampled_val["is_dev"] == 1].shape[0]) + "\t" + str(
    #     oversampled_val.loc[oversampled_val["is_dev"] == 0].shape[0]) + "\n")
    information += str("TEST:\t\t" + str(oversampled_test.shape[0]) + "\t" + str(
        oversampled_test.loc[oversampled_test["is_dev"] == 1].shape[0]) + "\t" + str(
        oversampled_test.loc[oversampled_test["is_dev"] == 0].shape[0]) + "\n")

    print(information)

    merge = oversampled_train_and_val.merge(oversampled_test, left_on=["author_id", "subreddit_id", "comment_id"],
                                            right_on=["author_id", "subreddit_id", "comment_id"])
    print(merge[(merge["take"] == 1) & (merge["is_dev"] == 1)])

    # ==================================
    # Save pickles for the Embedding Size and Unique user counts
    # ==================================
    v_img = 768
    with open(DATASET_NAME + "/IMGMODEL/data_10+10/V_TXT", "wb") as f:
        pickle.dump(v_img, f)

    n_usr = pd.unique(df["author_id"]).shape[0]
    with open(DATASET_NAME + "/IMGMODEL/data_10+10/N_USR", "wb") as f:
        pickle.dump(n_usr, f)