import pandas as pd

def userwise_random_oversampling(train):
    def balance_user(user_group):
        positives=user_group["Toxicity"].sum()
        negatives=len(user_group)-positives
        
        extra_samples = train[(~(train['subreddit_id'].isin(user_group['subreddit_id']))) & (train['Toxicity']==(negatives>positives))].sample(max(positives,negatives)-min(positives,negatives))
        
        return pd.concat([user_group,extra_samples])

    train = train.groupby(['author_id']).apply(balance_user).reset_index(drop=True)
    return train