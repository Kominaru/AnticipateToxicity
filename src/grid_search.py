import matplotlib.pyplot as plt
from src.training import train_toxicity_model
import numpy as np 
from os import makedirs
from src.models.mf import MF
from src.dataset import ToxicityCommentsDataset
from src.test import test_saved, test_roc_grid
from src.plots import plot_toxicity_distribution_grid

def save_params_text(path, *paramdicts):
    text = ""
    makedirs(path, exist_ok=True)
    with open(f"{path}/config.txt", "a") as f:
        for d in paramdicts:
            for param in d:
                f.write(f"{param} = {d[param]} \n")
                text+=f"{param} = {d[param]} \n"
    return text


    

def grid_search(params, name='test'):
    best_test_loss = 1e6
    text = save_params_text(f'grid_search/{name}',params)

    dataset = ToxicityCommentsDataset(filename=params['dataset_name'])
    dataset.split(params['split_strategy'])
    if 'sampling_strategy' in params:
        dataset.apply_negative_sampling(strategy= params['sampling_strategy'])

    print(np.average(dataset._y_train.cpu().detach().numpy()))
    # plot_toxicity_distribution_grid(f'grid_search/{name}',dataset.train)
    # plot_toxicity_distribution_grid(f'grid_search/{name}',dataset.train,dataset.test)


    for d in params['d']:
        i=1
        fig = plt.figure(figsize=(5+5*len(params['reg']),5*len(params['lr'])))
        for lr in params['lr']:
            for reg in params['reg']:
                
                if params['model_name'] == 'MF':
                    model = MF(dataset.nusers, dataset.nsubs, d=d, bias=params['bias'])
                results, min_test_loss = train_toxicity_model(model, dataset, learning_rate=lr, l2_reg=reg, epochs=params['epochs'], batch_size=params['batch_size'], best_test_loss=best_test_loss, figure_path=f'grid_search/{name}', weights=params['weights'])

                if min_test_loss<=best_test_loss:
                    best_test_loss=min_test_loss
                
                epochs_run=len(results["train_loss"])

                #Plot current training interation:
                plt.subplot(len(params['lr']),len(params['reg']),i)
                plt.title(f"d={d} | lr={lr} | l2-reg={reg}",fontdict={'fontsize': 12})

                plt.xticks(np.arange(0,params['epochs']+1,100),fontsize=12)
                plt.yticks(np.arange(0.25,1.5+0.25,0.25),fontsize=12)

                plt.xlabel("Epoch",fontsize=12)
                plt.ylabel("BCE Loss", fontsize=12)
                

                plt.xlim(0,params['epochs'])

                plt.ylim(0,1.5)
                
                plt.plot(np.arange(0,epochs_run,1),results["train_loss"], color="red",alpha=.25,label="Train Loss") #Train loss evolution
                plt.plot(np.arange(0,epochs_run,1),results["test_loss"], color="blue",alpha=.25,label="Test Loss") #Test loss evolution

                if i==1: plt.legend(loc="upper left")

                plt.twinx() #Swap axis

                plt.yticks(np.arange(30,101,10),fontsize=12)

                plt.ylabel("Metric performance (%)",fontsize=12)
                plt.ylim(30,100)

                plt.plot(np.arange(0,epochs_run,1),results["train_acc"], color="red", label="Train Acc (%)")         #Train acc evolution
                plt.plot(np.arange(0,epochs_run,1),results["test_acc"], color="blue", label="Test Acc (%)")   #Test acc evolution

                plt.plot(np.arange(0,epochs_run,1),results["train_recall"], '--', color="red", label="Train Recall (%)" , alpha=.3)   # Train TPR (%) evolution
                plt.plot(np.arange(0,epochs_run,1),results["test_recall"], '--', color="blue", label="Test Recall (%)", alpha=.3)   #Test TPR (%) evolution

                if i==1: plt.legend(loc="center left")
                i+=1

        plt.gcf().text(0.01, 0.95, text, fontsize=10, linespacing=1.5 , verticalalignment='top')
        plt.tight_layout()
        plt.gcf().subplots_adjust(left=5/(5+5*len(params['reg'])))
        plt.savefig(f"grid_search/{name}/d_{d}.pdf")

    test_saved(name, 'best-model.pt', dataset)
    test_roc_grid(name, 'best-model.pt', dataset)
    test_roc_grid(name, 'best-model.pt', dataset, use_train=True)