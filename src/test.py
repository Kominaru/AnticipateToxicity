from math import ceil, sqrt
from random import random, randrange, uniform
from matplotlib import ticker
from matplotlib.ticker import LogLocator
import torch
import pandas as pd
import numpy as np 
from src.utils import get_confmatrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
import seaborn as sns

def get_predictions_from_dataloader(test_dataloader,model_path):
    
    
    model = torch.load(model_path)
    model.eval()
    
    df = {"author_id":[],"subreddit_id":[],"output":[], "label":[], 'score':[]}

    for _, (inputs, labels) in enumerate(test_dataloader):
        
        labels = labels.round()

        scores = model.predict(inputs)
        predictions = ((scores.cpu() > 0.5)).float().detach().numpy()

        labels = labels.detach().cpu().numpy()
        inputs = inputs.detach().cpu().numpy()

        df['author_id']+=list(inputs[:,0])
        df['subreddit_id']+=list(inputs[:,1])
        df['output']+=list(predictions)
        df['label']+=list(labels)
        df['score']+=list(scores.cpu())

    #Generate dataframe of results for the test set. Information for each is (author_id,hit)
    df = pd.DataFrame.from_dict(df)
    return df


def plot_user_subreddit_contour(xx,grid_values,name,data_params,title):
    max_subreddit_toxicity = xx[np.max(np.where(np.nansum(grid_values,axis=1)>0))]
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(10,5))
    plt.contourf(xx,xx,grid_values,**data_params)
    plt.colorbar()
    plt.ylabel("Subreddit's mean\ntoxicity (Train)")
    plt.yticks([0,0.1,0.2,0.3])
    plt.xlabel("User's mean toxicity (Train)")
    plt.title(f"{title} by Train \n(user's mean toxicity, subreddit's mean toxicity)")
    plt.ylim(0,round(max_subreddit_toxicity,1))
    plt.tight_layout()
    plt.savefig(f"{name}",dpi=300)







def test_roc_grid(directory,model_name,dataset,use_train=False):

    def plot_roc_curve(ii,jj,labels,scores):
        
       

        print("Subplot", subreddit_tiles, user_tiles, user_tiles*ii+jj+1)
        plt.subplot(subreddit_tiles,user_tiles,user_tiles*ii+jj+1)
        plt.title(f'tox(s): ({ii*tile_size:.2f},{(ii+1)*tile_size:.2f}) | tox(u): ({jj*tile_size:.2f},{(jj+1)*tile_size:.2f})')

        plt.plot([0,1],[0,1])
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.xlim(0,1)
        plt.ylim(0,1)

        if len(labels)>0 and labels.sum()<len(labels):
            fpr, tpr, _ = roc_curve(labels,scores)
            plt.plot(fpr, tpr)
        else: 
            return np.nan

    def plot_toxicdist_hist(ii,jj,scores, predictions):
        plt.subplot(subreddit_tiles,user_tiles,user_tiles*ii+jj+1)
        plt.title(f'tox(s): ({ii*tile_size:.2f},{(ii+1)*tile_size:.2f}) | tox(u): ({jj*tile_size:.2f},{(jj+1)*tile_size:.2f})')

        plt.xlim(0,1)
        plt.hist2d(scores, predictions, bins=(20, 20), cmap="Blues", range=np.array([(0, 1), (0, 1)]), norm='log', vmin=1, vmax=10000)
        plt.xlabel('Toxicity Score')
        plt.ylabel('Toxicity Prediction')

    train_set = dataset.train
    test_set = dataset.test if use_train==False else dataset.train

    train_dataloader, test_dataloader = dataset.get_dataloaders()
    if use_train==True: test_dataloader=train_dataloader

    test_results = get_predictions_from_dataloader(test_dataloader, f"grid_search/{directory}/{model_name}")
    test_results = test_results.merge(test_set,on=['author_id','subreddit_id'],how='inner')
    print(test_results)

    user_mean_toxicity = train_set[train_set['author_id'].isin(test_set["author_id"])].groupby('author_id')['Toxicity'].mean().to_frame('mean_toxicity').reset_index()
    subreddit_mean_toxicity = train_set[train_set['subreddit_id'].isin(test_set["subreddit_id"])].groupby('subreddit_id')['Toxicity'].mean().to_frame('mean_toxicity').reset_index()

    tile_size=.1
    subreddit_tiles = ceil(subreddit_mean_toxicity['mean_toxicity'].max()/tile_size)
    user_tiles = ceil(user_mean_toxicity['mean_toxicity'].max()/tile_size)

    auc_grid = np.empty((subreddit_tiles,user_tiles))
    fig = plt.figure(figsize=(5*user_tiles,5*subreddit_tiles))
    plt.suptitle('ROC by mean user and subreddit toxicity')
    for ii,subreddit_toxicity in enumerate(np.linspace(0,tile_size*(subreddit_tiles-1),subreddit_tiles)):
        for jj,user_toxicity in enumerate(np.linspace(0,tile_size*(user_tiles-1),user_tiles)):

            current_users = user_mean_toxicity[user_mean_toxicity['mean_toxicity'].between(user_toxicity,user_toxicity+tile_size)]['author_id']
            current_subs = subreddit_mean_toxicity[(subreddit_mean_toxicity['mean_toxicity'].between(subreddit_toxicity,subreddit_toxicity+tile_size))]['subreddit_id']

            active_samples= test_results[(test_results["author_id"].isin(current_users)) & (test_results["subreddit_id"].isin(current_subs))]

            plot_roc_curve(ii,jj,active_samples['label'],active_samples['score'])

    


    plt.savefig(f"grid_search/{directory}/roc-grid-{'train' if train_set.equals(test_set) else 'test'}.pdf",dpi=300, bbox_inches = "tight")
    plt.clf()
    plt.cla()

    fig = plt.figure(figsize=(5*user_tiles,5*subreddit_tiles))
    plt.suptitle(f'(toxicity score, toxicity prediction) by mean user and subreddit toxicity ({"train" if train_set.equals(test_set) else "test"})')
    for ii,subreddit_toxicity in enumerate(np.linspace(0,tile_size*(subreddit_tiles-1),subreddit_tiles)):
        for jj,user_toxicity in enumerate(np.linspace(0,tile_size*(user_tiles-1),user_tiles)):

            current_users = user_mean_toxicity[user_mean_toxicity['mean_toxicity'].between(user_toxicity,user_toxicity+tile_size)]['author_id']
            current_subs = subreddit_mean_toxicity[(subreddit_mean_toxicity['mean_toxicity'].between(subreddit_toxicity,subreddit_toxicity+tile_size))]['subreddit_id']

            active_samples= test_results[(test_results["author_id"].isin(current_users)) & (test_results["subreddit_id"].isin(current_subs))]

            plot_toxicdist_hist(ii,jj,active_samples['Toxicbert_score'], active_samples['score'])


    plt.savefig(f"grid_search/{directory}/toxicity-distributions-grid-{'train' if train_set.equals(test_set) else 'test'}.pdf",dpi=300, bbox_inches = "tight")


def test_saved(directory,model_name,dataset):

    def initialize_square_grid(size):
        grid = np.empty((size,size))
        grid.fill(np.nan)
        return grid

    _, test_dataloader = dataset.get_dataloaders()
    
    test_results = get_predictions_from_dataloader(test_dataloader, f"grid_search/{directory}/{model_name}")

    def plot_tox_distribution(set,name):

        avg_user_toxicity = train_set.groupby('author_id')['Toxicity'].mean().reset_index(drop=False)
        avg_user_toxicity.columns = ['author_id','avg_u_tox']

        avg_sub_toxicity = train_set.groupby('subreddit_id')['Toxicity'].mean().reset_index(drop=False)
        avg_sub_toxicity.columns = ['subreddit_id','avg_s_tox']

        set = pd.merge(set,avg_user_toxicity,on= 'author_id', how='left')
        set = pd.merge(set,avg_sub_toxicity,on= 'subreddit_id', how='left')

        nbins=50
        xi = np.linspace(0,1,nbins)

        test_cases = np.empty((xi.shape[0],xi.shape[0]))
        print(test_cases)

        for ii,i in enumerate(xi):
            for jj,j in enumerate(xi):
                active_samples = set[set['avg_u_tox'].between(i-0.05,i+0.05) & set['avg_s_tox'].between(j-0.05,j+0.05)]
                test_cases[ii,jj] = len(active_samples)

        test_cases=test_cases.T
        # max_subreddit_toxicity = xi[np.max(np.where(np.nansum(test_cases,axis=1)>0))]

        # plt.rcParams.update({'font.size': 17})
        # plt.figure(figsize=(10,5))
        # plt.contourf(xi,xi,test_cases,cmap='Blues',locator=ticker.LogLocator())
        # plt.colorbar()
        # plt.ylabel("Subreddit's mean\ntoxicity")
        # plt.xlabel("User's mean toxicity")
        # plt.title(f"{name} sample distribution by \n (user's mean toxicity, subreddit's mean toxicity)")
        
        # plt.ylim(0,round(max_subreddit_toxicity,1))
        # plt.yticks([0,0.1,0.2,0.3])
        # plt.tight_layout()
        # plt.savefig(f"figuras_memoria/test_cases_{name}.pdf",dpi=300, bbox_inches = "tight")
        # plt.show()
        plot_user_subreddit_contour(xi,test_cases,f"grid_search/{directory}/meantoxicity-combs-best-model-{name}cases.pdf",data_params={"locator":LogLocator(),"cmap":"Blues"},title=f"{name} Cases")

    train_set, test_set = dataset.train, dataset.test

    plot_tox_distribution(train_set, name = 'Train')
    plot_tox_distribution(test_set, name = 'Test')

    user_mean_toxicity = train_set[train_set['author_id'].isin(test_set["author_id"])].groupby('author_id')['Toxicity'].mean().to_frame('mean_toxicity').reset_index()
    subreddit_mean_toxicity = train_set[train_set['subreddit_id'].isin(test_set["subreddit_id"])].groupby('subreddit_id')['Toxicity'].mean().to_frame('mean_toxicity').reset_index()

    samples_per_axis = 25
    window_size = .1

    def print_metrics(labels,preds):
        labels=labels.to_numpy()
        preds=preds.to_numpy()

        tp = np.sum(np.logical_and(preds,labels))
        tn = np.sum(np.logical_and(np.logical_not(preds),np.logical_not(labels)))
        fn = np.sum(np.logical_and(np.logical_not(preds),labels))
        fp = np.sum(np.logical_and(preds,np.logical_not(labels)))

        accuracy = (tp+tn)/(tn+fp+tp+fn)
        sensitivity = (tp)/(tp+fn)
        specificity = (tn)/(tn+fp)
        g_mean = sqrt(sensitivity*specificity)

        print(f"ACC\t{accuracy:.2f}\tSEN\t{sensitivity:.2f}\tSPE\t{specificity:.2f}\tG-M\t{g_mean:.2f}")

    print_metrics(test_results['label'],test_results['output'])

    xx = np.around(np.linspace(0,1,samples_per_axis),2)

    no_test_cases = initialize_square_grid(samples_per_axis)
    balanced_acc = initialize_square_grid(samples_per_axis)
    acc = initialize_square_grid(samples_per_axis)
    auc = initialize_square_grid(samples_per_axis) 

    for ii,subreddit_toxicity in enumerate(xx):
        for jj,user_toxicity in enumerate(xx):

            current_users = user_mean_toxicity[user_mean_toxicity['mean_toxicity'].between(user_toxicity-window_size,user_toxicity+window_size)]['author_id']
            current_subs = subreddit_mean_toxicity[(subreddit_mean_toxicity['mean_toxicity'].between(subreddit_toxicity-window_size,subreddit_toxicity+window_size))]['subreddit_id']

            active_samples= test_results[(test_results["author_id"].isin(current_users)) & (test_results["subreddit_id"].isin(current_subs))]

            

            if len(active_samples)>0:
                hits = get_confmatrix(active_samples['label'],active_samples['output'])
                
                tp = hits['tp']
                fp = hits['fp']
                tn = hits['tn']
                fn = hits['fn']

                no_test_cases[ii,jj]=len(active_samples)
                balanced_acc[ii,jj] = np.nanmean([(tp)/(tp+fn),(tn)/(tn+fp)])
                acc[ii,jj] = (tp+tn)/len(active_samples)
                if len(active_samples)>active_samples['label'].sum() and active_samples['label'].sum()>0:

                    auc[ii,jj] = roc_auc_score(active_samples['label'],active_samples['score'])


    plot_user_subreddit_contour(xx,no_test_cases,f"grid_search/{directory}/meantoxicity-combs-best-model-testcases.pdf",data_params={"locator":LogLocator(),"cmap":"Blues"},title="Test Cases")
    plot_user_subreddit_contour(xx,balanced_acc,f"grid_search/{directory}/meantoxicity-combs-best-model-balancedacc.pdf",data_params={"levels":np.arange(0,1+0.1,0.1),"cmap":"Spectral"}, title="Balanced Accuracy")
    plot_user_subreddit_contour(xx,acc,f"grid_search/{directory}/meantoxicity-combs-best-model-acc.pdf",data_params={"levels":np.arange(0,1+0.1,0.1),"cmap":"RdYlGn"}, title="Accuracy")
    plot_user_subreddit_contour(xx,auc,f"grid_search/{directory}/meantoxicity-combs-best-model-auc.pdf",data_params={"levels":np.arange(0,1+0.1,0.1),"cmap":"RdYlGn"}, title="Test AUC-ROC")