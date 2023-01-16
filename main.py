from src.dataset import ToxicityCommentsDataset
from src.models.mf import MF
from src.training import train_toxicity_model
from src.grid_search import grid_search
params = {
    'mode':'grid_search',

    #Experimental setup
    'dataset_name':'coronavirus_v1',
    'model_name':'MF',
    'split_strategy':'classwise_leave_one_out',
    'weights': 'class',
    'bias':'prior',

    #Grid Search params. If mode=='train', it will pick the first element of each list
    'd': [64,256,1024],
    'lr':[1e-3,1e-4,1e-5],
    'reg':[1e-2,5e-4,0],
    'batch_size': 2**12,
    'epochs':500
}

if params['mode']=='train':

    # model=MF(dataset.nusers, dataset.nsubs, 256).to('cuda')
    # metrics = train_toxicity_model(model, dataset, params)
    pass

elif params['mode']=='grid_search':
    grid_search(params,name='mf_prior')





