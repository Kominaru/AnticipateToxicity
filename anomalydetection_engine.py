
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

from sklearn.svm import OneClassSVM

from src.dataset import ToxicityCommentsDataset

# Load dataset
dataset_name='coronavirus_v1'
dataset = ToxicityCommentsDataset(filename=dataset_name)
dataset.split('classwise_leave_one_out')

train, test = dataset.train.sample(frac = 1), dataset.test.sample(frac = 1)
comments = dataset.comments

print("Loading BERT embeddings...")

# Load the BERT embeddings to obtain the sample features to input to the model. 
# Right now for every (user,subreddit) sample we're using a concatenation of: 
# - average embedding of the comments of train interactions of the user
# - max embedding of the comments of train interactions of the user
# - average embedding of the comments of train interactions of the subreddit
# - max embedding of the comments of train interactions of the subreddit

bertavg_users = np.zeros((dataset.nusers,768*2))
bertavg_subreddits = np.zeros((dataset.nsubs,768*2))

comments_train = comments[~(comments[['author_id','subreddit_id']].apply(tuple,axis=1).isin(test[['author_id','subreddit_id']].apply(tuple,axis=1)))]
bert_embeds=pickle.load(open(f"BERT_EMBEDDINGS/{dataset_name}",'rb'))

commentlists_user=comments_train.groupby('author_id')['comment_id'].apply(list).reset_index(name="comment_ids")
bertavg_users[commentlists_user["author_id"].to_list(),:768]=np.stack(commentlists_user["comment_ids"].apply(lambda x: np.average(bert_embeds[x,:],axis=0)).to_numpy(),axis=0)
bertavg_users[commentlists_user["author_id"].to_list(),768:]=np.stack(commentlists_user["comment_ids"].apply(lambda x: np.max(bert_embeds[x,:],axis=0)).to_numpy(),axis=0)

commentlists_subreddit=comments_train.groupby('subreddit_id')['comment_id'].apply(list).reset_index(name="comment_ids")
bertavg_subreddits[commentlists_subreddit["subreddit_id"].to_list(),:768]=np.stack(commentlists_subreddit["comment_ids"].apply(lambda x: np.average(bert_embeds[x,:],axis=0)).to_numpy(),axis=0)
bertavg_subreddits[commentlists_subreddit["subreddit_id"].to_list(),768:]=np.stack(commentlists_subreddit["comment_ids"].apply(lambda x: np.max(bert_embeds[x,:],axis=0)).to_numpy(),axis=0)

#Contamination (toxicity) ratio in the dataset (unsupervised training)
toxics_ratio = train['Toxicity'].sum()/len(train['Toxicity'])
print(f"Using a contamination ratio of {toxics_ratio}")

no_trees=100
features_ratio =1
samples_per_tree= 1.0
features_mode = "mul"

# model = IsolationForest(random_state=42, n_estimators=no_trees, contamination=toxics_ratio, max_samples=samples_per_tree, verbose=2, max_features=features_ratio)
model = OneClassSVM(nu=toxics_ratio, kernel="rbf", gamma='auto', cache_size=1600, verbose=2, shrinking=False)

print("Creating samples...")

if features_mode == "mul":
    X_train = bertavg_users[train['author_id']]*bertavg_subreddits[train['subreddit_id']]
else: 
    X_train = np.concatenate((bertavg_users[train['author_id']],bertavg_subreddits[train['subreddit_id']]),axis=1)
y_train = train['Toxicity']

if features_mode == "mul":
    X_test = bertavg_users[test['author_id']]*bertavg_subreddits[test['subreddit_id']]
else: 
    X_test = np.concatenate((bertavg_users[test['author_id']],bertavg_subreddits[test['subreddit_id']]),axis=1)
y_test = test['Toxicity']

print(f"Training model from {X_train.shape[0]} samples with {X_train.shape[1]} features...")
y_pred = model.fit(X_train)

print(f"Predicting on {X_train.shape[0]} train samples with {X_test.shape[1]} features...")
y_pred = model.predict(X_train)
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

print('Confussion matrix (train):')
print(confusion_matrix(y_train, y_pred))

print(f"Predicting on {X_test.shape[0]} test samples with {X_test.shape[1]} features...")
y_pred = model.predict(X_test)
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1


print('Confussion matrix (test)')
print(confusion_matrix(y_test, y_pred))

## ISOLATION FOREST
# Confussion matrix (train):
# [[123764  10306]
#  [ 10306   2656]]
# Predicting on 28499 test samples with 1536 features...
# Confussion matrix (test)
# [[20330  4178]
#  [ 3199   792]]

## OneClassSVM
# Confussion matrix (train):
# [[123112  10958]
#  [ 10957   2002]]
# Predicting on 28502 test samples with 1536 features...
# Confussion matrix (test)
# [[20905  3603]
#  [ 3472   522]]