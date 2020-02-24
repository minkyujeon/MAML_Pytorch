import torch
import sklearn
import sklearn.datasets
from torch.utils.data import Dataset
from torch.utils.data import  random_split

from utils import ratio2nsamples

class CustomDataset(Dataset):
    def __init__(self, size=2000, num_classes=3, n_features=20, n_informative=15):
        X,y = sklearn.datasets.make_classification(n_samples=size,n_features=n_features,
                                           n_informative=n_informative, n_classes=num_classes)
        self.X = torch.from_numpy(X).type(torch.FloatTensor)
        self.y = torch.from_numpy(y).type(torch.FloatTensor)

    def __getitem__(self,index):
        return self.X[index] , self.y[index].long()

    def __len__(self):
        return len(self.X)


def generate_dataset(szdata, szbatch, num_classes,trts_ratio,n_features):
    dataset = CustomDataset(size=szdata, num_classes=num_classes,n_features=n_features)
    train_dataset, test_dataset = random_split(dataset, ratio2nsamples(trts_ratio, szdata))
    return train_dataset,test_dataset