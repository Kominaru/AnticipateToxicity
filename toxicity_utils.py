import pandas as pd

#
# Define and perform train/test split, generate datasets
#

def train_test_split(df, method, drop_untested_users=False, balance_train=False, max_imbalance_ratio=2):

    ## 1.- Creation of the TEST and TRAIN sets ##
    user_groups=df.groupby('author_id')
    subreddit_groups=df.groupby('subreddit_id')

    test=[]

    # Approach "random": For each user with >=10 interactions, add 10% of their interactions to test set
    if method=="random":
        for _,group in user_groups:
            if group.shape[0]>=10:
                test+=(group.sample(n=int(group.shape[0]*0.15)).to_dict(orient="records"))

    # Approach "valid_users": For each user with >=10 interactions and >=2 interactions of each type, add one interac. of each type to test set
    elif method=="valid_users":
        for _,group in user_groups:
            if group.shape[0]>=10 and group["Toxicity"].sum()>1 and ((group["Toxicity"].shape[0]-group["Toxicity"].sum())>1):
                test += (group[group["Toxicity"]==1].sample(1).to_dict(orient="records"))
                test += (group[group["Toxicity"]==0].sample(1).to_dict(orient="records"))

    # Approach "valid_subreddits": For each subreddit with >=10 interactions and >=2 interactions of each type, add one interac. of each type to test set
    elif method=="valid_subreddits":
        for _,group in subreddit_groups:
            if group.shape[0]>=10 and group["Toxicity"].sum()>1 and ((group["Toxicity"].shape[0]-group["Toxicity"].sum())>1):
                test += (group[group["Toxicity"]==1].sample(1).to_dict(orient="records"))
                test += (group[group["Toxicity"]==0].sample(1).to_dict(orient="records"))

    # Approach "valid_both": Select all the samples that match the conditions from "valid_users" AND "valid_subreddits". Choose from that selection randomly
    # this doesn't ensure that the test set won't have all samples available for a certain (user,toxicity) or (subreddit,toxicity) combination. 
    elif method=="valid_both":
        
        valid_users=[]
        valid_subreddits=[]
        for user,group in user_groups:
            if group.shape[0]>=10 and group["Toxicity"].sum()>2 and ((group["Toxicity"].shape[0]-group["Toxicity"].sum())>2):
                valid_users.append(user)


        for subreddit,group in subreddit_groups:
            if group.shape[0]>=10 and group["Toxicity"].sum()>2 and ((group["Toxicity"].shape[0]-group["Toxicity"].sum())>2):
                valid_subreddits.append(subreddit)
        
        print(f"Found {len(valid_users)} users  and {len(valid_subreddits)} that meet criteria")

        valid_rows = df[(df["author_id"].isin(valid_users)) & (df["subreddit_id"].isin(valid_subreddits))]

        print(f"Intersecting these users and subreddits, {valid_rows['author_id'].nunique()} and {valid_rows['subreddit_id'].nunique()} are preserved")

        test += (valid_rows[valid_rows["Toxicity"]==1].sample(valid_rows["author_id"].nunique()).to_dict(orient="records"))
        test += (valid_rows[valid_rows["Toxicity"]==0].sample(valid_rows["author_id"].nunique()).to_dict(orient="records"))

    # Approach "controlled_users": Select all the samples that match the conditions from "valid_users" AND "valid_subreddits". Then select exactly
    # ONE sample per each (user,toxicity) combination. Doesn't ensure that the tes set won't have of samples available for a certain (subreddit, toxicity) combination
    elif method=="controlled_users":
        valid_users=[]
        valid_subreddits=[]
        for user,group in user_groups:
            if group.shape[0]>=10 and group["Toxicity"].sum()>2 and ((group["Toxicity"].shape[0]-group["Toxicity"].sum())>2):
                valid_users.append(user)
        for subreddit,group in subreddit_groups:
            if group.shape[0]>=10 and group["Toxicity"].sum()>5 and ((group["Toxicity"].shape[0]-group["Toxicity"].sum())>5):
                valid_subreddits.append(subreddit)
        
        print(f"Found {len(valid_users)} users  and {len(valid_subreddits)} that meet criteria")
        
        test = df[(df["author_id"].isin(valid_users)) & (df["subreddit_id"].isin(valid_subreddits))]

        test = test.groupby(['author_id','Toxicity']).apply(lambda g: g.sample(min(1,g.shape[0]))).reset_index(drop=True)

        test = test.to_dict(orient='records')


    test = pd.DataFrame(test)
    print(f"Total test samples: {test.shape[0]}")

    train = pd.concat([df, test]).drop_duplicates(keep=False)
    
    ## 2.- Balancing of the TRAIN set ## 
    
    if drop_untested_users:
        train=train[(train["author_id"].isin(test["author_id"])) & (train["subreddit_id"].isin(test["subreddit_id"]))]

    if balance_train in ["user","global"]:
        newtrain=[]
        if balance_train=="user":
            user_groups=train.groupby('author_id')
            subreddit_groups=train.groupby('subreddit_id')

            #For each user, choose all the samples (n samples) from the minority class, and a maximum of n*max_imbalance_ratio from the majority class
            #This means each user's training samples won't be class-imbalanced by >max_imbalanced ratio in any case

            for user,group in user_groups:
                positives = group["Toxicity"].sum()
                negatives = group["Toxicity"].shape[0]-group["Toxicity"].sum()

                majorclass_max = int(round(min(positives,negatives)*max_imbalance_ratio)) 

                newtrain += group[group["Toxicity"]==1].sample(min(positives,majorclass_max)).to_dict(orient="records")
                newtrain += group[group["Toxicity"]==0].sample(min(negatives,majorclass_max)).to_dict(orient="records")

        if balance_train=="global":
            user_groups=train.groupby('author_id')
            subreddit_groups=train.groupby('subreddit_id')

            #Add one negative and one positive sample per user to the train set (if they have them)
            for user,group in user_groups:
                positives = group["Toxicity"].sum()
                negatives = group["Toxicity"].shape[0]-group["Toxicity"].sum()

                newtrain += group[group["Toxicity"]==1].sample(min(positives, negatives,1)).to_dict(orient="records")
                newtrain += group[group["Toxicity"]==0].sample(min(positives, negatives,1)).to_dict(orient="records")
            temptrain = pd.DataFrame(newtrain)

            #Check what subreddits are represented in the train set so far
            temp_positive = temptrain[temptrain["Toxicity"]==1]["subreddit_id"].to_list()
            temp_negative = temptrain[temptrain["Toxicity"]==0]["subreddit_id"].to_list()

            #If a subreddit still has no positive or negative sample on train set, add it (if they have them)
            for subreddit,group in subreddit_groups:
                positives = group["Toxicity"].sum()
                negatives = group["Toxicity"].shape[0]-group["Toxicity"].sum()
                if subreddit not in temp_positive:
                    newtrain += group[group["Toxicity"]==1].sample(min(positives, negatives,1)).to_dict(orient="records")
                if subreddit not in temp_negative:
                    newtrain += group[group["Toxicity"]==0].sample(min(positives, negatives,1)).to_dict(orient="records")

            temptrain = pd.DataFrame(newtrain)
            spare = pd.concat([train, temptrain]).drop_duplicates(keep=False)

            #Assuming the minority class c has a total of n samples not in test, choose samples from the current "spare"
            #samples so each class ends up with n samples. This means the classes should be globally balanced even if they're not
            #balanced per user. 

            #The resulting train set should also have at least one positive and one negative sample for each user or subreddit 
            #appearing in the test set

            positives = temptrain["Toxicity"].sum()
            negatives = temptrain["Toxicity"].shape[0]-temptrain["Toxicity"].sum()
            
            max_positives=train["Toxicity"].sum()
            max_negatives=train["Toxicity"].shape[0] - train["Toxicity"].sum()

            print(positives, negatives, max_positives, max_negatives)

            newtrain += spare[spare["Toxicity"]==1].sample(min(max_positives,max_negatives)-positives).to_dict(orient="records")
            newtrain += spare[spare["Toxicity"]==0].sample(min(max_positives,max_negatives)-negatives).to_dict(orient="records")

        train = pd.DataFrame(newtrain)
        if train.shape[0]<100000:
            train = train.sample(100000,replace=True)
        # train.groupby('subreddit_id')["Toxicity"].mean().plot.hist()
    return train, test


from torch.utils.data import DataLoader, Dataset

class ToxicityDataset(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.n_samples=y.size(0)
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.n_samples

def get_dataloader(x,y,batch_size):
    return DataLoader(dataset=ToxicityDataset(x,y),batch_size=batch_size,shuffle=True)
            