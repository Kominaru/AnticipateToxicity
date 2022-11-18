from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#
# Define and perform train/test split, generate datasets
#

def train_test_split(df, method, drop_untested_users=False, balance_train=False, max_imbalance_ratio=2, test_activity_threshold=2, directory_path=""):

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
            if group["Toxicity"].sum()>=test_activity_threshold and ((group["Toxicity"].shape[0]-group["Toxicity"].sum())>=test_activity_threshold):
                valid_users.append(user)


        for subreddit,group in subreddit_groups:
            if group["Toxicity"].sum()>5 and ((group["Toxicity"].shape[0]-group["Toxicity"].sum())>5):
                valid_subreddits.append(subreddit)
        
        print(f"Found {len(valid_users)} users  and {len(valid_subreddits)} that meet criteria")

        valid_rows = df[(df["author_id"].isin(valid_users)) & (df["subreddit_id"].isin(valid_subreddits))]

        print(f"Intersecting these users and subreddits, {valid_rows['author_id'].nunique()} and {valid_rows['subreddit_id'].nunique()} are preserved")

        test += (valid_rows[valid_rows["Toxicity"]==1].sample(valid_rows["author_id"].nunique()).to_dict(orient="records"))
        test += (valid_rows[valid_rows["Toxicity"]==0].sample(valid_rows["author_id"].nunique()).to_dict(orient="records"))

    # Approach "controlled_users": Select all the samples that match the conditions from "valid_users" AND "valid_subreddits". Then select exactly
    # ONE sample per each (user,toxicity) combination. Doesn't ensure that the test set won't have of samples available for a certain (subreddit, toxicity) combination
    elif method=="controlled_users":
        valid_users=[]
        valid_subreddits=[]
        for user,group in user_groups:
            if group["Toxicity"].sum()>=test_activity_threshold and ((group["Toxicity"].shape[0]-group["Toxicity"].sum())>=test_activity_threshold):
                valid_users.append(user)
        for subreddit,group in subreddit_groups:
            if group["Toxicity"].sum()>5 and ((group["Toxicity"].shape[0]-group["Toxicity"].sum())>5):
                valid_subreddits.append(subreddit)
        
        print(f"Found {len(valid_users)} users  and {len(valid_subreddits)} that meet criteria")
        
        test = df[(df["author_id"].isin(valid_users)) & (df["subreddit_id"].isin(valid_subreddits))]

        test = test.groupby(['author_id','Toxicity']).apply(lambda g: g.sample(min(1,g.shape[0]))).reset_index(drop=True)

        test = test.to_dict(orient='records')

    elif method == "valid_combs":

        # For each (user_id, toxicity) combination, if there's >=2 samples with it, reserve one for test set
        test = df.groupby(['author_id','Toxicity']).apply(lambda x: x.sample(max(min(2,len(x))-1,0))).reset_index(drop=True)

        #Obtain the sample count for each (subreddit,toxicity) combination
        subreddit_combs_counts = df.groupby(['subreddit_id','Toxicity']).size().to_frame('size').to_dict()['size']
        
        #Ensure we're not selecting all the n samples of any (subreddit, toxicity) combination, selecting n-1 at most
        test = test.groupby(['subreddit_id','Toxicity']).apply(lambda x: x.sample(min(len(x),subreddit_combs_counts[x.name]-1))).reset_index(drop=True)

        #Balance the test set by undersampling the majority class (which should be the non-toxic comments)
        # test = test.groupby('Toxicity').apply(lambda x: x.sample(min(len(x),test['Toxicity'].value_counts().min()))).reset_index(drop=True)
        test = test.to_dict(orient='records')

    elif method == "leave_one_out":

        dfactive = df.groupby('subreddit_id').filter(lambda x: len(x)>=2)
        dfactive = dfactive.groupby('author_id').filter(lambda x: len(x)>=2)

        test = dfactive.groupby('author_id').apply(lambda x: x.sample(1))
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
            user_groups=train.groupby(['author_id','Toxicity'])
            subreddit_groups=train.groupby(['subreddit_id', 'Toxicity'])

            #Add at most two negative and two positive samples per user to the train set (if they have them)
            #This ensures all tested users have in train set the samples they were selected for
            for groupname,group in user_groups:
                newtrain += group.sample(min(2, len(group))).to_dict(orient="records")

            temptrain = pd.DataFrame(newtrain)

            #Check what subreddits are represented in the train set so far
            temp_samples_per_subreddit = temptrain.groupby(['subreddit_id','Toxicity'])

            #Forces tested subreddits to have the required samples in train set
            for groupname,group in temp_samples_per_subreddit:
                ogtrain_group = subreddit_groups.get_group(groupname)
                newtrain += ogtrain_group.sample(min(len(ogtrain_group)-len(group),5-len(group))).to_dict(orient="records")

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

    if balance_train=="negative_sampling":
        def negative_sampling(user_group):
            positives=user_group["Toxicity"].sum()
            negatives=len(user_group)-positives
            extra_samples = train[(~(train['subreddit_id'].isin(user_group['subreddit_id']))) & (train['Toxicity']==(negatives>positives))].sample(max(positives,negatives)-min(positives,negatives))
            # extra_samples['Toxicity']=min(positives,negatives)
            
            return pd.concat([user_group,extra_samples])
        train = train.groupby(['author_id']).apply(negative_sampling).reset_index(drop=True)

    
    print_sampledistribution_meantoxicity(train,test)
    # if train.shape[0]<100000:
    #     train = train.sample(100000,replace=True)


    

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
            
def print_sampledistribution_meantoxicity(train,test):

    user_mean_toxicity = train[train['author_id'].isin(test["author_id"])].groupby('author_id')['Toxicity'].mean().to_frame('mean_toxicity').reset_index()
    interac_per_author_train = train["author_id"].value_counts().reset_index()
    interac_per_author_train.columns=['author_id','count']

    xx = []
    acc = []
    tpr = []
    no_test_cases=[]
    positive_test_cases=[]
    negative_test_cases=[]
    tnr=[]

    for x in np.arange(0,1+0.01,0.01):
        toxicity_users = user_mean_toxicity[(user_mean_toxicity['mean_toxicity']>=(x-0.05)) & (user_mean_toxicity['mean_toxicity']<=(x+0.05)) & user_mean_toxicity['author_id'].isin(interac_per_author_train[interac_per_author_train["count"]>10]["author_id"])]['author_id']

        active_samples = test[test['author_id'].isin(toxicity_users)]
        no_test_cases.append(active_samples.shape[0])
        positive_test_cases.append(len(active_samples[active_samples['Toxicity']==1]))
        negative_test_cases.append(len(active_samples[active_samples['Toxicity']==0]))

        xx.append(x)
    

    plt.cla()
    plt.clf()
    plt.xlabel("Users with x+-0.1 mean toxicity")
    plt.plot(xx,no_test_cases, color="green", label="Test Cases", alpha=.3)
    plt.plot(xx,positive_test_cases,'--', color="green", label="Positive Test Cases", alpha=.5)
    plt.plot(xx,negative_test_cases, ':',color="green", label="Negative Test Cases", alpha=.5)

    plt.ylabel("No. of Test Cases")
    plt.yscale("log")
    plt.title("performance by mean interaction toxicity")
    plt.tight_layout()
    plt.show()

def obtain_tox_user_weights(train_set):
    def obtain_weight(user_group):
        positives = (user_group["Toxicity"]>.5).sum()
        return 1-(positives/len(user_group))
    
    weights = train_set.groupby(['author_id']).apply(obtain_weight).reset_index(drop=True).to_numpy()
    return weights

def get_userweight_array(inputs,labels,tox_user_weights):
    weights = tox_user_weights[inputs]
    weights = labels*weights + np.logical_not(labels)*(1-weights)
    return weights

def obtain_tox_grid_weights(train_set):

    user_mean_toxicity = train_set.groupby('author_id')['Toxicity'].mean().to_frame('mean_toxicity').reset_index()
    subreddit_mean_toxicity = train_set.groupby('subreddit_id')['Toxicity'].mean().to_frame('mean_toxicity').reset_index()

    samples_per_axis = 25
    window_size = .1

    xx = np.around(np.linspace(0,1,samples_per_axis+1),2)

    weights = np.empty((xx.shape[0],xx.shape[0],2))
    weights.fill(np.nan)

    for ii,subreddit_toxicity in enumerate(xx):
        for jj,user_toxicity in enumerate(xx):
            toxicity_users = user_mean_toxicity[user_mean_toxicity['mean_toxicity'].between(user_toxicity-window_size,user_toxicity+window_size)]['author_id']
            toxicity_subs = subreddit_mean_toxicity[subreddit_mean_toxicity['mean_toxicity'].between(subreddit_toxicity-window_size,subreddit_toxicity+window_size)]['subreddit_id']


            #Get test samples of those active users
            active_samples= train_set [(train_set["author_id"].isin(toxicity_users)) & (train_set["subreddit_id"].isin(toxicity_subs)) ]

            #Get the acc obtained in those samples


            if len(active_samples)>0:
                weights[ii,jj,0]=1-(len(active_samples)-active_samples['Toxicity'].sum())/len(train_set)
                weights[ii,jj,1]=1-(active_samples['Toxicity'].sum())/len(train_set)
    
    find_nearest = np.vectorize(lambda value: (np.abs(xx - value)).argmin())
    user_mean_toxicity = find_nearest(user_mean_toxicity['mean_toxicity'])
    subreddit_mean_toxicity = find_nearest(subreddit_mean_toxicity['mean_toxicity'])
    weights*=(1/np.nanmin(weights))

    plt.contourf(xx,xx,weights[...,0])
    plt.colorbar()
    plt.show()

    plt.contourf(xx,xx,weights[...,1])
    plt.colorbar()
    plt.show()

    return user_mean_toxicity,subreddit_mean_toxicity,weights

def get_gridweight_array(inputs,labels,user_mean_toxicity,sub_mean_toxicity,grid_weights):
    weights = grid_weights[user_mean_toxicity[inputs[:,0]],sub_mean_toxicity[inputs[:,1]],labels.astype(np.int8)]
    
    print(user_mean_toxicity[inputs[:,0][np.where(np.isnan(weights))]],sub_mean_toxicity[inputs[:,1][np.where(np.isnan(weights))]])
    return weights
