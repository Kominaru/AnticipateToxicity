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
    'data_labelling':'binary',
    'two_step_training':False,

    #Grid Search params. If mode=='train', it will pick the first element of each list
    'd': [61],
    'lr':[1e-4],
    'reg':[5e-4],
    'batch_size': 2**14,
    'epochs':500
}

if params['mode']=='train':

    # model=MF(dataset.nusers, dataset.nsubs, 256).to('cuda')
    # metrics = train_toxicity_model(model, dataset, params)
    pass

elif params['mode']=='grid_search':
    grid_search(params,name='memoria_figuras')





