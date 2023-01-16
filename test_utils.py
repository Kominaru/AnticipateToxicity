from math import sqrt
from matplotlib import pyplot as plt
from numpy import int16
import pandas as pd
import torch
import numpy as np
from os import remove
import seaborn as sns
from torchmetrics import MeanSquaredError
from matplotlib.ticker import LogLocator
from src.utils import get_confmatrix

def test(test_dataloader,model_name, CONFIG):
    
    
    model = torch.load(model_name)
    model.eval()

    authors=[]
    outputs=[]
    labels=[]
    subreddits=[]

    for _, (test_inputs, test_labels) in enumerate(test_dataloader):
        
        outputs_test = torch.squeeze(model(test_inputs)).cpu()
        if CONFIG["TRAINING_GOAL"]!="regression":
            outputs_test = (outputs_test > 0.0)
        outputs_test=outputs_test.detach().numpy()
        labels_test = test_labels.detach().cpu().numpy()

        outputs+=list(outputs_test)
        labels+=list(labels_test)
        authors+=list(test_inputs[:,0].detach().cpu().numpy())
        subreddits+=list(test_inputs[:,1].detach().cpu().numpy())

    #Generate dataframe of results for the test set. Information for each is (author_id,hit)
    df = pd.DataFrame.from_dict({"author_id":authors,"subreddit_id":subreddits,"output":outputs, "label": labels})
    return df

def test_best_model(directory_path,comments,interactions,test_set,train_set,test_dataloader,epochs, CONFIG):


    train_set=train_set.drop_duplicates(keep='first')
    train_set=train_set[train_set[['author_id','subreddit_id']].apply(tuple,axis=1).isin(interactions[['author_id','subreddit_id']].apply(tuple,axis=1))]
    interac_per_author_train = train_set["author_id"].value_counts().reset_index()

    not_test_interacc = interactions[~(interactions[['author_id','subreddit_id']].apply(tuple,axis=1).isin(test_set[['author_id','subreddit_id']].apply(tuple,axis=1)))]
    not_test_comments = comments[~(comments[['author_id','subreddit_id']].apply(tuple,axis=1).isin(test_set[['author_id','subreddit_id']].apply(tuple,axis=1)))]

    interac_per_author_nottest = not_test_interacc["author_id"].value_counts().reset_index()
    comment_per_author_nottest = not_test_comments["author_id"].value_counts().reset_index()

    interac_per_author_train.columns = ['author_id', 'count']
    interac_per_author_nottest.columns = ['author_id', 'count']
    comment_per_author_nottest.columns = ['author_id', 'count']


    df = test(test_dataloader,f"{directory_path}/best-model.pt", CONFIG)

    def plot_activity(activity_counts,df,name):

        xx=[]

        acc=[]
        tpr=[]
        tnr=[]

        no_test_cases=[]

        for i in range(0,100):

            #Count number of active users
            active_users=activity_counts[activity_counts["count"]>i]["author_id"]
            if len(active_users)==0:
                break

            #Get test samples of those active users
            active_samples= df [df["author_id"].isin(active_users)]

            #Get the acc obtained in those samples
            tp  = np.float32(len(active_samples[(active_samples["label"]==1) & (active_samples["output"]==1)]))
            fp  = np.float32(len(active_samples[(active_samples["label"]==1) & (active_samples["output"]==0)]))
            fn  = np.float32(len(active_samples[(active_samples["label"]==0) & (active_samples["output"]==1)]))
            tn  = np.float32(len(active_samples[(active_samples["label"]==0) & (active_samples["output"]==0)]))

            active_acc = (tp+tn)/(tp+tn+fp+fn)

            active_tpr = tp/(tp+fn)
            active_tnr = tn/(tn+fp)

            xx.append(i)

            acc.append(active_acc)
            tpr.append(active_tpr)
            tnr.append(active_tnr)

            no_test_cases.append(active_samples.shape[0])



        
        plt.cla()
        plt.clf()
        plt.plot(xx,acc, color="blue", label="Test Acc (%)")
        plt.plot(xx,tpr,'--',color="navy", label="Test TPR (%)" , alpha=.5)
        plt.plot(xx,tnr,':',color="navy", label="Test TNR (%)" , alpha=.5)
        plt.xlabel("Users with >=x activity")
        plt.ylabel("Metric Performance")
        plt.ylim(0.4,1)
        plt.legend()
        plt.twinx()
        plt.plot(xx,no_test_cases, color="green", label="Test Cases", alpha=.3)
        plt.ylabel("No. of Test Cases")
        plt.yscale("log")
        plt.title(name)
        plt.legend()
        plt.savefig(f"{directory_path}/{name}.pdf")
        plt.show()


    
    if CONFIG["TRAINING_GOAL"]!="regression":
        plot_activity(interac_per_author_nottest,df,"performance by user nontest interactions")
        plot_activity(interac_per_author_train,df,"performance by user train interactions")
        plot_activity(comment_per_author_nottest, df, "performance by user nontest comments")

    plot_performance_by_mean_comb_toxicity(train_set,test_set,f"{directory_path}/best-model.pt",test_dataloader,directory_path,interac_per_author_train,CONFIG)

    # for i in range (0,epochs,100):
    #     plot_performance_by_mean_user_toxicity(train_set,test_set,f"{directory_path}/last-model-epoch-{i}.pt",test_dataloader,directory_path,i,interac_per_author_train)
    # plot_performance_by_mean_user_toxicity(train_set,test_set,f"{directory_path}/best-model.pt",test_dataloader,directory_path,i,interac_per_author_train)



def plot_performance_by_mean_user_toxicity(train_set,test_set,model_name,test_dataloader,directory_path,i,interac_per_author_train, CONFIG):
    
    df = test(test_dataloader,model_name, CONFIG)

    user_mean_toxicity = train_set[train_set['author_id'].isin(test_set["author_id"])].groupby('author_id')['Toxicity'].mean().to_frame('mean_toxicity').reset_index()

    print("hello")
    print(user_mean_toxicity)

    xx = []
    acc = []
    tpr = []
    no_test_cases=[]
    positive_test_cases=[]
    negative_test_cases=[]
    tnr=[]

    for x in np.arange(0,1+0.01,0.01):
        toxicity_users = user_mean_toxicity[(user_mean_toxicity['mean_toxicity']>=(x-0.05)) & (user_mean_toxicity['mean_toxicity']<=(x+0.05)) & user_mean_toxicity['author_id'].isin(interac_per_author_train[interac_per_author_train["count"]>=5]["author_id"])]['author_id']



        #Get test samples of those active users
        active_samples= df [df["author_id"].isin(toxicity_users)]

        #Get the acc obtained in those samples
        tp  = np.float32(len(active_samples[(active_samples["label"]==1) & (active_samples["output"]==1)]))
        fp  = np.float32(len(active_samples[(active_samples["label"]==0) & (active_samples["output"]==1)]))
        fn  = np.float32(len(active_samples[(active_samples["label"]==1) & (active_samples["output"]==0)]))
        tn  = np.float32(len(active_samples[(active_samples["label"]==0) & (active_samples["output"]==0)]))

        active_acc = (tp+tn)/(tp+tn+fp+fn)

        active_tpr = tp/(tp+fn)
        active_tnr = tn/(tn+fp)

        xx.append(x)

        acc.append(active_acc)
        tpr.append(active_tpr)
        tnr.append(active_tnr)
        
        no_test_cases.append(active_samples.shape[0])
        positive_test_cases.append(tp+fn)
        negative_test_cases.append(tn+fp)


        print(x, active_acc, active_tpr, active_samples.shape[0])


    plt.cla()
    plt.clf()
    plt.plot(xx,acc, color="blue", label="Test Acc (%)")
    plt.plot(xx,tpr,'--',color="navy", label="Test Sensitivity (%)" , alpha=.5)
    plt.plot(xx,tnr,':',color="navy", label="Test Specificity (%)" , alpha=.5)
    plt.xlabel("Users with x+-0.1 mean toxicity")
    plt.ylabel("Metric Performance")
    plt.ylim(0,1)
    plt.legend(bbox_to_anchor=(1.2, 0.7))
    plt.twinx()
    plt.plot(xx,no_test_cases, color="green", label="Test Cases", alpha=.3)
    # plt.plot(xx,positive_test_cases,'--', color="green", label="Positive Test Cases", alpha=.5)
    # plt.plot(xx,negative_test_cases, ':',color="green", label="Negative Test Cases", alpha=.5)

    plt.ylabel("No. of Test Cases")
    plt.yscale("log")
    plt.title("performance by mean interaction toxicity")
    plt.legend(bbox_to_anchor=(2, 1))
    plt.savefig(f"{directory_path}/meantoxicity-epoch{i}.pdf",dpi=300, bbox_inches = "tight")
    plt.show()

def plot_performance_by_mean_comb_toxicity(train_set,test_set,model_name,test_dataloader,directory_path,interac_per_author_train,CONFIG):
    
    df = test(test_dataloader,model_name,CONFIG)

    user_mean_toxicity = train_set[train_set['author_id'].isin(test_set["author_id"])].groupby('author_id')['Toxicity'].mean().to_frame('mean_toxicity').reset_index()
    subreddit_mean_toxicity = train_set[train_set['subreddit_id'].isin(test_set["subreddit_id"])].groupby('subreddit_id')['Toxicity'].mean().to_frame('mean_toxicity').reset_index()

    print(subreddit_mean_toxicity['mean_toxicity'].max())

    samples_per_axis = 25
    window_size = .1

    xx = np.around(np.linspace(0,1,samples_per_axis+1),2)

    acc = np.empty((xx.shape[0],xx.shape[0]))
    acc.fill(np.nan)
    no_test_cases = np.empty((xx.shape[0],xx.shape[0]))
    no_test_cases.fill(np.nan)

    rmse = np.empty((xx.shape[0],xx.shape[0]))
    rmse.fill(np.nan)

    balanced_acc = np.empty((xx.shape[0],xx.shape[0]))
    balanced_acc.fill(np.nan)
    
    for ii,subreddit_toxicity in enumerate(xx):
        for jj,user_toxicity in enumerate(xx):
            toxicity_users = user_mean_toxicity[(user_mean_toxicity['mean_toxicity'].between(user_toxicity-window_size,user_toxicity+window_size)) & (user_mean_toxicity['author_id'].isin(interac_per_author_train[interac_per_author_train["count"]>=5]["author_id"]))]['author_id']
            toxicity_subs = subreddit_mean_toxicity[(subreddit_mean_toxicity['mean_toxicity'].between(subreddit_toxicity-window_size,subreddit_toxicity+window_size))]['subreddit_id']


            #Get test samples of those active users
            active_samples= df [(df["author_id"].isin(toxicity_users)) & (df["subreddit_id"].isin(toxicity_subs)) ]

            #Get the acc obtained in those samples
            


            if len(active_samples)>0:
                no_test_cases[ii,jj]=len(active_samples)
                acc[ii,jj] = (tp+tn)/(tp+tn+fp+fn)             #accuracy
                if CONFIG["TRAINING_GOAL"]=="regression":
                    criterion = MeanSquaredError(squared=False)
                    rmse[ii,jj] = criterion(torch.Tensor(active_samples["label"]),torch.Tensor(active_samples["output"]))

                balanced_acc[ii,jj] = np.nanmean([(tp)/(tp+fn),(tn)/(tn+fp)]) #balanced acuracy

    plot_user_subreddit_contour(xx,acc,f"{directory_path}/meantoxicity-combs-best-model-acc.pdf",data_params={"levels":np.arange(0,1+0.1,0.1),"cmap":"Spectral"},title="Accuracy")
    plot_user_subreddit_contour(xx,no_test_cases,f"{directory_path}/meantoxicity-combs-best-model-testcases.pdf",data_params={"locator":LogLocator(),"cmap":"Spectral"},title="Test Cases")

    plot_user_subreddit_contour(xx,balanced_acc,f"{directory_path}/meantoxicity-combs-best-model-balancedacc.pdf",data_params={"levels":np.arange(0,1+0.1,0.1),"cmap":"Spectral"}, title="Balanced Accuracy")


    # if CONFIG["TRAINING_GOAL"]=="regression":
    #     plt.subplots(figsize=(10,10))
    #     # heatmap = sns.heatmap(no_test_cases,cmap="YlGnBu",vmin=0,norm=LogNorm(),xticklabels=xx, yticklabels=xx, annot=True, annot_kws={"fontsize":8})
    #     # heatmap.invert_yaxis()

    #     plt.contourf(xx,xx,rmse,locator=LogLocator(),cmap="coolwarm")
    #     plt.ylabel("Mean user toxicity")
    #     plt.xlabel("Mean subreddit toxicity")
    #     plt.title("RMSE by mean user and subreddit toxicity")

    #     plt.savefig(f"{directory_path}/meantoxicity-combs-best-model-rmse.pdf",dpi=300, bbox_inches = "tight")
    #     plt.show()


def plot_user_subreddit_contour(xx,grid_values,name,data_params,title):
    max_subreddit_toxicity = xx[np.max(np.where(np.nansum(grid_values,axis=1)>0))]
    plt.figure(figsize=(10,10*max_subreddit_toxicity))
    plt.contourf(xx,xx,grid_values,**data_params)
    plt.colorbar()
    plt.ylabel("Mean subreddit toxicity")
    plt.xlabel("Mean user toxicity")
    plt.title(f"{title} by mean user and subreddit toxicity")
    plt.ylim(0,round(max_subreddit_toxicity,1))
    plt.savefig(f"{name}",dpi=300, bbox_inches = "tight")
    plt.show()
