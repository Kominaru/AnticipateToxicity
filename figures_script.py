from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor, sigmoid

df = pd.read_csv("datasets/coronavirus_v1.csv")

unique_interactions = df.drop_duplicates(["author_id", "subreddit_id"], keep="first")

plt.rcParams.update({"font.size": 25, "figure.figsize": [9, 6]})

subreddits_per_author = unique_interactions.groupby("author_id").size()
subreddits_per_author[subreddits_per_author > 20] = 20

ax = subreddits_per_author.plot.hist(
    bins=range(0, 21, 2), xticks=np.arange(0, 21, 5), yticks=(0, 1000, 2000, 3000, 4000, 5000, 6000), xlim=(1, 20)
)
ax.set_xlabel("Num. of subreddits \n visited by the user")
ax.set_ylabel("Num. of users")
ax.xaxis.set_major_formatter(lambda x, pos: f"{x}" if x < 20 else ">20")
plt.tight_layout()
plt.savefig("figuras_memoria/subs_per_user.pdf")
plt.cla()
plt.clf()

authors_per_subreddit = unique_interactions.groupby("subreddit_id").size()
authors_per_subreddit[authors_per_subreddit > 1000] = 1000

ax = authors_per_subreddit.plot.hist(
    bins=range(0, 1001, 100), xticks=np.arange(0, 1001, 250), yticks=(0, 25, 50, 75, 100), xlim=(50, 1000)
)
ax.set_xlabel("Num. of users who comment \n in the subreddit")
ax.set_ylabel("Num. of subreddits")
ax.xaxis.set_major_formatter(lambda x, pos: f"{x}" if x < 1000 else ">1000")
plt.tight_layout()
plt.savefig("figuras_memoria/users_per_sub.pdf")

plt.cla()
plt.clf()


df["Toxicity"] = sigmoid(Tensor(df["Toxicity"]))

df = df.groupby(["author_id", "subreddit_id"], as_index=False)["Toxicity"].mean()

# Make figure flatter
plt.rcParams.update({"font.size": 18, "figure.figsize": [9, 2.5]})

ax = df["Toxicity"].plot.hist(
    bins=np.linspace(0, 1, 11),
    xticks=np.linspace(0, 1, 11),
    xlim=(0, 1),
    yticks=[1, 10, 100, 1000, 10000, 100000],
    ylim=(1000, 200000),
    color="#b96f74",
    figsize=(9, 4),
)
ax.set_xlabel("Average toxicity of (user, subreddit) interaction")
ax.set_ylabel("No. of Interactions")
ax.set_yscale("log")
plt.tight_layout()
plt.savefig("figuras_memoria/toxicidad_por_interaccion.pdf")
plt.cla()
plt.clf()

exit()

# from src.dataset import ToxicityCommentsDataset
# from src.splitting import classwise_leave_one_out

# df = ToxicityCommentsDataset(filename='coronavirus_v1')

# df = df.df

# train_val, test = classwise_leave_one_out(df)
# train, val = classwise_leave_one_out(train_val)


# print(len(train), len(val), len(test))
# print(len(train)+ len(val)+ len(test))

# print(df['author_id'].nunique(), df['subreddit_id'].nunique(), df['Toxicity'].sum(), len(df)-df['Toxicity'].sum())
# print(train['author_id'].nunique(), train['subreddit_id'].nunique(), train['Toxicity'].sum(), len(train)-train['Toxicity'].sum())
# print(val['author_id'].nunique(), val['subreddit_id'].nunique(), val['Toxicity'].sum(), len(val)-val['Toxicity'].sum())
# print(test['author_id'].nunique(), test['subreddit_id'].nunique(), test['Toxicity'].sum(), len(test)-test['Toxicity'].sum())


##BASELINES

# def print_metrics(labels,preds):
#     labels=labels.to_numpy()
#     preds=preds.to_numpy()

#     tp = np.sum(np.logical_and(preds,labels))
#     tn = np.sum(np.logical_and(np.logical_not(preds),np.logical_not(labels)))
#     fn = np.sum(np.logical_and(np.logical_not(preds),labels))
#     fp = np.sum(np.logical_and(preds,np.logical_not(labels)))

#     accuracy = (tp+tn)/(tn+fp+tp+fn)
#     sensitivity = (tp)/(tp+fn)
#     specificity = (tn)/(tn+fp)
#     g_mean = sqrt(sensitivity*specificity)

#     print(f"ACC\t{accuracy:.3f}\tSEN\t{sensitivity:.3f}\tSPE\t{specificity:.3f}\tG-M\t{g_mean:.3f}")

# #BASELINE NON

# test['NON']=0

# print_metrics(test['Toxicity'],test['NON'])

# #BASELINE RND

# test['RND'] = list(np.random.binomial(1,train['Toxicity'].mean(),(len(test))))

# print_metrics(test['Toxicity'],test['RND'])

# #BASELINE USR

# avg_user_toxicity = train.groupby('author_id')['Toxicity'].mean().reset_index(drop=False)
# avg_user_toxicity.columns = ['author_id','USR']

# test = pd.merge(test,avg_user_toxicity,on='author_id',how='left')
# test['USR'] = test['USR'].apply(lambda p : np.random.binomial(1,p))

# print_metrics(test['Toxicity'],test['USR'])

# #BASELINE AVG

# avg_user_toxicity = train.groupby('author_id')['Toxicity'].mean().reset_index(drop=False)
# avg_user_toxicity.columns = ['author_id','avg_u_tox']

# avg_sub_toxicity = train.groupby('subreddit_id')['Toxicity'].mean().reset_index(drop=False)
# avg_sub_toxicity.columns = ['subreddit_id','avg_s_tox']

# test = pd.merge(test,avg_user_toxicity,on= 'author_id', how='left')
# test = pd.merge(test,avg_sub_toxicity,on= 'subreddit_id', how='left')

# test['AVG'] = test.apply(lambda row : np.random.binomial(1,(row['avg_u_tox']+row['avg_s_tox'])/2),axis=1)

# print_metrics(test['Toxicity'],test['AVG'])

# train=train_val

# avg_user_toxicity = train.groupby('author_id')['Toxicity'].mean().reset_index(drop=False)
# avg_user_toxicity.columns = ['author_id','avg_u_tox']

# avg_sub_toxicity = train.groupby('subreddit_id')['Toxicity'].mean().reset_index(drop=False)
# avg_sub_toxicity.columns = ['subreddit_id','avg_s_tox']

# test = pd.merge(test,avg_user_toxicity,on= 'author_id', how='left')
# test = pd.merge(test,avg_sub_toxicity,on= 'subreddit_id', how='left')

# train = pd.merge(train,avg_user_toxicity,on= 'author_id', how='left')
# train = pd.merge(train,avg_sub_toxicity,on= 'subreddit_id', how='left')

# print(len(test))

# import seaborn as sns
# from scipy.stats import kde
# from matplotlib import ticker, cm

# def plot_tox_distribution(set,name):
#     nbins=50
#     xi = np.linspace(0,1,nbins)

#     test_cases = np.empty((xi.shape[0],xi.shape[0]))
#     print(test_cases)

#     for ii,i in enumerate(xi):
#         for jj,j in enumerate(xi):
#             active_samples = set[set['avg_u_tox'].between(i-0.05,i+0.05) & set['avg_s_tox'].between(j-0.05,j+0.05)]
#             test_cases[ii,jj] = len(active_samples)

#     test_cases=test_cases.T
#     max_subreddit_toxicity = xi[np.max(np.where(np.nansum(test_cases,axis=1)>0))]

#     plt.rcParams.update({'font.size': 17})
#     plt.figure(figsize=(10,5))
#     plt.contourf(xi,xi,test_cases,cmap='Blues',locator=ticker.LogLocator())
#     plt.colorbar()
#     plt.ylabel("Subreddit's mean\ntoxicity")
#     plt.xlabel("User's mean toxicity")
#     plt.title(f"{name} sample distribution by \n (user's mean toxicity, subreddit's mean toxicity)")
#     plt.ylim(0,round(max_subreddit_toxicity,1))
#     plt.yticks([0,0.1,0.2,0.3])
#     plt.tight_layout()
#     plt.savefig(f"figuras_memoria/test_cases_{name}.pdf",dpi=300, bbox_inches = "tight")
#     plt.show()

# plot_tox_distribution(test,"Test")
# plot_tox_distribution(train,"Train")
