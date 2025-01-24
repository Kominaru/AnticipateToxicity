from math import ceil
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torch import Tensor, sigmoid
from torch.utils.data import DataLoader, Dataset
from src import splitting, negative_sampling

class ToxicityPartition(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.n_samples=y.size(0)
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.n_samples

class ToxicityCommentsDataset(Dataset):
    def __init__(self, filename, directory='datasets/', int_def='mean', task='classification', device='cuda'):
        self.name = filename
        
        df=pd.read_csv(f'{directory}/{filename}.csv',encoding='UTF_8')

        # Original toxicity values are logits so they must be passed through sigmoid to convert to (0,1) range
        df["Toxicity"]=sigmoid(Tensor(df["Toxicity"])) 
        self.comments = df

        #How to characterise an interaction: mean toxicity of max toxicity of its comments
        if int_def=="max":
            df=df.groupby(['author_id','subreddit_id'],as_index=False)['Toxicity'].max()
        elif int_def=="mean":
            df=df.groupby(['author_id','subreddit_id'],as_index=False)['Toxicity'].mean()

        #Store raw dataframe with probability scores
        self.df=df

        print(f"Loaded dataset with {len(df)} interactions")

        #If we're not doing regression, converts interactions to binary: 0 (nontoxic) or 1 (toxic)
        if task!="regression":   
            df['Toxicbert_score']=df['Toxicity']
            df['Toxicity']=df['Toxicity'].apply(lambda x: round(x))
        
        self.data=df
        self.nusers=df['author_id'].nunique()
        self.nsubs=df['author_id'].nunique()

        self.device=device
    

    # Performs inplace splitting of the Dataset into train and test sets, also creating the tensors for
    # training and testing 
    def split(self,strategy='classwise_leave_one_out'):

        # Classwise leave-one-out: for each (user,toxicity) combination with more than 2 samples, reserve
        # one sample for testing
        if strategy=='classwise_leave_one_out':
            train, test = splitting.classwise_leave_one_out(self.data)
        elif strategy=='classwise_leave_one_out_requireboth':
            train, test = splitting.classwise_leave_one_out_requireboth(self.data)
        elif strategy == 'classwise_user_holdout':
            train, test = splitting.classwise_user_holdout(self.data)

        print (f'Obtained {len(train)} train samples and {len(test)} test samples')

        #Removing non-toxic users
        train = train.groupby('author_id').filter(lambda a: 0<=a['Toxicity'].mean()<=1)
        test = test[test['author_id'].isin(train['author_id'])]
        test = test[test['subreddit_id'].isin(train['subreddit_id'])]


        # Removing inactive users
        # train = train.groupby('author_id').filter(lambda a: len(a)>=5)
        # test = test[test['author_id'].isin(train['author_id'])]
        # test = test[test['subreddit_id'].isin(train['subreddit_id'])]
        
        
        print (f'Obtained {len(train)} train samples and {len(test)} test samples')

        self.train = train
        self.test = test

        self._create_train_test_tensors(train=self.train, test=self.test)
    
    def apply_negative_sampling(self,strategy='userwise_ROS'):

        if strategy == 'userwise_ROS':
            train = negative_sampling.userwise_random_oversampling(train=self.train)
            self._create_train_test_tensors(train=train)
    
    def get_toxicclass_weight(self,strategy='binary', set='train'):
        if set=='train':
            aux_set = self.train
        elif set =='test':
            aux_set = self.test
        
        if strategy=='binary' or set =='test':
                return (len(aux_set)-aux_set['Toxicity'].sum())/aux_set['Toxicity'].sum()

        elif strategy=='bernouilli' and set == 'train':
                return (1-aux_set['Toxicbert_score'].mean())/aux_set['Toxicbert_score'].mean()
        

        
    # Creates the Pytorch tensors (train inputs, train outputs, test inputs, test outputs) for the splitted dataset.

    def load_BERT_embeddings(self):
        bertavg_users = np.zeros((self.nusers,768*2))
        bertavg_subreddits = np.zeros((self.nsubs,768*2))

        comments = comments[~(comments[['author_id','subreddit_id']].apply(tuple,axis=1).isin(self.test[['author_id','subreddit_id']].apply(tuple,axis=1)))]

        import pickle
        from torch import Tensor

        embeddings=pickle.load(open(f"BERT_EMBEDDINGS/{self.name}",'rb'))

        commentlists_user=comments.groupby('author_id')['comment_id'].apply(list).reset_index(name="comment_ids")
        bertavg_users[commentlists_user["author_id"].to_list(),:768]=np.stack(commentlists_user["comment_ids"].apply(lambda x: np.average(embeddings[x,:],axis=0)).to_numpy(),axis=0)
        bertavg_users[commentlists_user["author_id"].to_list(),768:]=np.stack(commentlists_user["comment_ids"].apply(lambda x: np.max(embeddings[x,:],axis=0)).to_numpy(),axis=0)

        commentlists_subreddit=comments.groupby('subreddit_id')['comment_id'].apply(list).reset_index(name="comment_ids")
        bertavg_subreddits[commentlists_subreddit["subreddit_id"].to_list(),:768]=np.stack(commentlists_subreddit["comment_ids"].apply(lambda x: np.average(embeddings[x,:],axis=0)).to_numpy(),axis=0)
        bertavg_subreddits[commentlists_subreddit["subreddit_id"].to_list(),768:]=np.stack(commentlists_subreddit["comment_ids"].apply(lambda x: np.max(embeddings[x,:],axis=0)).to_numpy(),axis=0)
        
        self.bert_users=Tensor(bertavg_users).float().to(self.device)
        self.bert_subs=Tensor(bertavg_subreddits).float().to(self.device)

    def _create_train_test_tensors(self, train = None, test = None):
        
        
        

        if train is not None:
            X_train=train[['author_id','subreddit_id']].to_numpy().astype(int)
            y_train=train['Toxicbert_score'].to_numpy()
            self._X_train = Tensor(X_train).int().to(self.device)
            self._y_train = Tensor(y_train).float().to(self.device)

            # Compute weight of the minoritary class samples for binary classification

        if test is not None: 
            X_test=test[['author_id','subreddit_id']].to_numpy().astype(int)
            y_test=test['Toxicbert_score'].to_numpy()
            self._X_test = Tensor(X_test).int().to(self.device)
            self._y_test = Tensor(y_test).float().to(self.device)

       
    def get_dataloaders(self,batch_size=2**10):

            return (DataLoader(ToxicityPartition(self._X_train,self._y_train), batch_size=batch_size, shuffle=True),
                DataLoader(ToxicityPartition(self._X_test,self._y_test), batch_size=batch_size, shuffle=True))

        
