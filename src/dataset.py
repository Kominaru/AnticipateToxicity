import pandas as pd

class ToxicityCommentsDataset():
    def __init__(self, filename, directory='datasets/'):
        df=pd.read_csv(f'{directory}/{filename}.csv',encoding='UTF_8')