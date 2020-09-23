from omniglotNShot import OmniglotNShot
from meta import MetaLearner

import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self, n_way, img_size):
        super(Net, self).__init__()
        
        self.net = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3),
                                nn.AvgPool2d(kernel_size=2),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                
                                nn.Conv2d(64, 64, kernel_size=3),
                                nn.AvgPool2d(kernel_size=2),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                
                                nn.Conv2d(64, 64, kernel_size=3),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                
                                nn.Conv2d(64, 64, kernel_size=3),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
        
        self.fc = nn.Sequential(nn.Linear(64, 64),
                               nn.ReLU(inplace=True),
                               nn.Linear(64, n_way))
        
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x, target):
        # x:[5, 1, 28, 28] : 5 way 1 shot
        x = self.net(x)
        x = x.view(-1, 64)
        pred = self.fc(x)
        loss = self.loss(pred, target)
        
        return loss, pred
    
def main():
    meta_batch_size = 32
    n_way = 5
    k_shot = 1
    k_query = 1
    meta_lr = 1e-3
    num_updates = 5
    
    img_size = 28
    omni_data = OmniglotNShot('dataset', batch_size=meta_batch_size, n_way=n_way,
                             k_shot=k_shot, k_query=k_query, img_size=img_size)
    
    meta = MetaLearner(Net, (n_way, img_size), n_way=n_way, k_shot=k_shot, meta_batch_size=meta_batch_size,
                       alpha=0.1, beta=meta_lr, num_updates=num_updates).cuda()
    
    for episode_num in range(100):
        support_x, support_y, query_x, query_y = omni_data.get_batch('train') # support, query for train
        # support_x : [32, 5, 1, 28, 28]
        support_x = torch.from_numpy(support_x).float().cuda()
        query_x = torch.from_numpy(query_x).float().cuda()
        support_y = torch.from_numpy(support_y).long().cuda()
        query_y = torch.from_numpy(query_y).long().cuda()
        accs = meta(support_x, support_y, query_x, query_y)
        train_acc = np.array(accs).mean()
        
        if episode_num % 30 == 0:
            test_accs = []

            support_x, support_y, query_x, query_y = omni_data.get_batch('test') # support, query for test
            support_x = torch.from_numpy(support_x).float().cuda()
            query_x = torch.from_numpy(query_x).float().cuda()
            support_y = torch.from_numpy(support_y).long().cuda()
            query_y = torch.from_numpy(query_y).long().cuda()

            test_acc = meta.pred(support_x, support_y, query_x, query_y)
            test_accs.append(test_acc)

            test_acc = np.array(test_accs).mean()
            print('episode:',episode_num, '\tfintune acc:%.6f' % train_acc, '\t\ttest acc:%.6f' % test_acc)
    
if __name__ == '__main__':
    main()