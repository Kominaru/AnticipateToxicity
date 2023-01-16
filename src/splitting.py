import pandas as pd

def classwise_leave_one_out(df):
     # For each (user_id, toxicity) combination, if there's >=2 samples with it, reserve one for test set
    test = df.groupby(['author_id','Toxicity']).apply(lambda x: x.sample(1) if len(x)>=2 else x.sample(0)).reset_index(drop=True)

    #Obtain the sample count for each (subreddit,toxicity) combination
    subreddit_combs_counts = df.groupby(['subreddit_id','Toxicity']).size().to_frame('size').to_dict()['size']
    
    #Ensure we're not selecting all the n samples of any (subreddit, toxicity) combination, selecting n-1 at most
    test = test.groupby(['subreddit_id','Toxicity']).apply(lambda x: x.sample(min(len(x),subreddit_combs_counts[x.name]-1))).reset_index(drop=True)

    train = pd.concat([df, test]).drop_duplicates(keep=False)

    return train,test