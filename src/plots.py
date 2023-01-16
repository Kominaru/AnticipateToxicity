from math import ceil
from matplotlib import pyplot as plt
import numpy as np


def plot_toxicity_distribution_grid(directory, train, test=None):
        
        def plot_toxicdist_hist(ii,jj,scores):
            print("Subplot", subreddit_tiles, user_tiles, user_tiles*ii+jj+1)
            plt.subplot(subreddit_tiles,user_tiles,user_tiles*ii+jj+1)
            plt.title(f'tox(s): ({ii*tile_size:.2f},{(ii+1)*tile_size:.2f}) | tox(u): ({jj*tile_size:.2f},{(jj+1)*tile_size:.2f})')

            plt.xlim(0,1)
            plt.ylim(1,10000)
            plt.hist(scores, log=True, bins=20)

        if test is None:
            test=train

        
        user_mean_toxicity = train[train['author_id'].isin(test["author_id"])].groupby('author_id')['Toxicity'].mean().to_frame('mean_toxicity').reset_index()
        subreddit_mean_toxicity = train[train['subreddit_id'].isin(test["subreddit_id"])].groupby('subreddit_id')['Toxicity'].mean().to_frame('mean_toxicity').reset_index()

        tile_size=.1
        subreddit_tiles = ceil(subreddit_mean_toxicity['mean_toxicity'].max()/tile_size)
        user_tiles = ceil(user_mean_toxicity['mean_toxicity'].max()/tile_size)

        fig = plt.figure(figsize=(5*user_tiles,5*subreddit_tiles))
        plt.suptitle('ROC by mean user and subreddit toxicity')
        for ii,subreddit_toxicity in enumerate(np.linspace(0,tile_size*(subreddit_tiles-1),subreddit_tiles)):
            for jj,user_toxicity in enumerate(np.linspace(0,tile_size*(user_tiles-1),user_tiles)):

                current_users = user_mean_toxicity[user_mean_toxicity['mean_toxicity'].between(user_toxicity,user_toxicity+tile_size)]['author_id']
                current_subs = subreddit_mean_toxicity[(subreddit_mean_toxicity['mean_toxicity'].between(subreddit_toxicity,subreddit_toxicity+tile_size))]['subreddit_id']

                active_samples= test[(test["author_id"].isin(current_users)) & (test["subreddit_id"].isin(current_subs))]

                plot_toxicdist_hist(ii,jj,active_samples['Toxicbert_score'])


        plt.savefig(f"{directory}/toxicity-distribution-{'train' if train.equals(test) else 'test'}.pdf",dpi=300, bbox_inches = "tight")


        