import numpy as np

def get_confmatrix(labels,predictions):
    tp = np.sum(np.logical_and(predictions,labels))
    tn = np.sum(np.logical_and(np.logical_not(predictions),np.logical_not(labels)))
    fn = np.sum(np.logical_and(np.logical_not(predictions),labels))
    fp = np.sum(np.logical_and(predictions,np.logical_not(labels)))
    return {'tp':tp, 'tn':tn, 'fn':fn, 'fp':fp}
