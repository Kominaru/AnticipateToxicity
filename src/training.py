#
# Define the training cycle for the model
#
from math import sqrt
from torchviz import make_dot
from torch.optim import Adam
import torch
import numpy as np
from collections import Counter
from src.utils import get_confmatrix

def print_epoch(metrics, last=False):
    print(f"{len(metrics['train_loss'])}\t{metrics['train_loss'][-1]:.7f}\t{metrics['test_loss'][-1]:.7f}\t{metrics['train_acc'][-1]:.2f}\t\t{metrics['test_acc'][-1]:.2f}\t{metrics['train_recall'][-1]:.2f}\t\t{metrics['test_recall'][-1]:.2f}",end="\r")

def train_toxicity_model(model, dataset, learning_rate=1e-3, l2_reg=0, epochs=10, batch_size= 2**12, figure_path='grid_search/test', weights='class', best_test_loss=1e6, custom_loss='bce'):
    patience = 20
    trigger_times = 0

    def bce_loss(scores, labels, set='train'):
        """Binary Cross-Entropy (BCE) pointwise loss, also known as log loss or logistic loss.
        Args:
            scores (tensor): Tensor containing predictions for both positive and negative items.
            ratings (tensor): Tensor containing ratings for both positive and negative items.
        Returns:
            loss.
        """
        # Calculate Binary Cross Entropy loss
        if weights is None:
            criterion = torch.nn.BCELoss()
        elif weights=='class':
            if set=='train':
                criterion = torch.nn.BCELoss(weight=(labels*(dataset.train_tox_weight-1)+1).to(dataset.device))
            elif set=='test':
                criterion = torch.nn.BCELoss(weight=(labels*(dataset.test_tox_weight-1)+1).to(dataset.device))
        # elif weights=='grid':
        #         criterion = torch.nn.BCELoss(weight=get_grid_weights(labels)))
        
        loss = criterion(scores, labels)
        return loss

    # def rmse_loss(scores,labels,set='train'):
        
    #     if set=='train':
    #         weight=(labels*(dataset.train_tox_weight-1)+1).to(dataset.device)
    #     elif set=='test':
    #         weight=(labels*(dataset.test_tox_weight-1)+1).to(dataset.device)
        
    #     return sqrt(sum((scores-labels)^2*weight)/sum(weight))
    
    print(f"Training | {model} | L.Rate:{learning_rate} | L2 Reg:{l2_reg} | previous best={best_test_loss}")

    train_data, test_data = dataset.get_dataloaders(batch_size=batch_size)

    metrics={
        'train_loss':[],
        'test_loss':[],
        'train_acc':[],
        'test_acc':[],
        'train_recall':[],
        'test_recall':[]
    }

    #Model configuration
    optimizer=Adam(model.parameters(), learning_rate) 
    
    inputs,_ = next(iter(train_data))
    # print(inputs.device)
    # print(model.device)
    yhat = model(inputs)
    make_dot(yhat, params=dict(list(model.named_parameters()))).render(f"{figure_path}/rnn_torchviz", format="png")

    print("EPOCH\tLOSS_TRAIN\tLOSS_TEST\tACC_TRAIN\tACC_TEST\tTPR_TRAIN\tTPR_TEST")

    for epoch in range(int(epochs)):

        ## TRAINING PART ## 

        model.train()

        loss=0

        hits = Counter({'tp':0, 'tn':0, 'fn':0, 'fp':0})

        #Iterate training over train batches
        for _,(inputs,labels) in enumerate(train_data):

            optimizer.zero_grad()  # Setting our stored gradients equal to zero
            scores, regularizer = model.forward(inputs)

            if custom_loss=='bce':
                batch_loss = bce_loss(scores, labels, set='train')   
            # elif custom_loss=='rmse':
            #     batch_loss = rmse_loss(scores, labels, set='train')

            loss += batch_loss.item()*labels.size(0)

            batch_loss = batch_loss + l2_reg * regularizer
            batch_loss.backward()
            
            optimizer.step()

            labels = labels.detach().cpu().numpy()
            predictions = ((scores.cpu() > 0.5)).float().detach().numpy()

            hits+=get_confmatrix(labels,predictions)


        tp = hits['tp']
        fp = hits['fp']
        tn = hits['tn']
        fn = hits['fn']

        metrics['train_loss'].append(loss / len(train_data.dataset))
                
        metrics['train_acc'].append(100 * (tp+tn) / len(train_data.dataset))
        metrics['train_recall'].append(100*tp/(tp+fn))
        
    
           
        with torch.no_grad():
            # Compute metrics for test dataset
            model.eval()

            loss=0

            hits = Counter({'tp':0, 'tn':0, 'fn':0, 'fp':0})

            #Iterate training over train batches
            for _,(inputs,labels) in enumerate(test_data):

                scores = model.predict(inputs)
                
                if custom_loss=='bce':
                    batch_loss = bce_loss(scores, labels, set='test')   
                # elif custom_loss=='rmse':
                #     batch_loss = rmse_loss(scores, labels, set='test')
                batch_loss = batch_loss + l2_reg * regularizer

                loss += batch_loss.item()*labels.size(0)

                labels = labels.detach().cpu().numpy()
                predictions = ((scores.cpu() > 0.5)).float().detach().numpy()

                hits+=get_confmatrix(labels,predictions)

            tp = hits['tp']
            fp = hits['fp']
            tn = hits['tn']
            fn = hits['fn']

            metrics['test_loss'].append(loss / len(test_data.dataset))
                    
            metrics['test_acc'].append(100 * (tp+tn) / len(test_data.dataset))
            metrics['test_recall'].append(100*tp/(tp+fn))

            if epoch>=patience and metrics['test_loss'][-1]>=(np.average(metrics['test_loss'][-patience:-1])-1e-3):
                trigger_times+=1
                if trigger_times>=patience:
                    break
            else: 
                trigger_times=0

            if metrics['test_loss'][-1]<best_test_loss:

                best_test_loss=metrics['test_loss'][-1]
                torch.save(model,f"{figure_path}/best-model.pt")
                text=""
                with open(f"{figure_path}/best-params.txt","w") as f:
                    vars={"d":model.d,"reg":l2_reg,"lr":learning_rate,"epoch":epoch,"test_loss":metrics['test_loss'][-1]}
                    for param in vars:
                        text+=f"{param} = {vars[param]} \n"
                    f.write(text)

            print_epoch(metrics)
    
    print_epoch(metrics,last=True)
    return metrics, best_test_loss

