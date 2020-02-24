import shutil
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import torch
import torch.nn as nn
import numpy as np

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

def calc_confusion_mat(val_loader, model, args):
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cf = confusion_matrix(all_targets, all_preds).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt

    print('Class Accuracy : ')
    print(cls_acc)
    classes = [str(x) for x in args.cls_num_list]
    plot_confusion_matrix(all_targets, all_preds, classes)
    plt.savefig(os.path.join(args.root_log, args.store_name, 'confusion_matrix.png'))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def prepare_folders(args):
    
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(args, state, is_best):
    
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def multiclass_uniform_distribution(num_classes, num_samples, device):
    y = torch.randint(num_classes, size=(num_samples,)) 
    # The confidence score on the max dim is ranged from 1/K to 1:w
    yhat_max = (1-1/num_classes)*torch.rand(size=(num_samples,)).to(device)+1/num_classes 
    #yhat_max = torch.rand(size=(num_samples,)).to(device)
    y = y.to(device)
    y_onehot = nn.functional.one_hot(y, num_classes=num_classes).float().to(device)
    y_hat = torch.rand(num_samples,num_classes).to(device)
    y_hat_nm = torch.mul(1-y_onehot, y_hat)
    y_hat_nm = y_hat_nm / torch.sum(y_hat_nm,dim=1, keepdim=True)
    y_hat = y_hat_nm*(1-yhat_max[:,None])+y_onehot*yhat_max[:,None]
    return y, y_hat

def myrandint(N, wo=[]):
    while True:
        x = np.random.randint(N)
        if not (x in wo):
            return x

def multiclass_uniform_distribution_with_acc(num_classes, num_samples, device, eacc=0):
    y = torch.randint(num_classes, size=(num_samples,))
    # The confidence score on the max dim is ranged from 1/K to 1:w
    yhat_max = (1-1/num_classes)*torch.rand(size=(num_samples,)).to(device)+1/num_classes 
    y = y.to(device)
    y_onehot = nn.functional.one_hot(y, num_classes=num_classes).float().to(device)
    y_hat = torch.rand(num_samples,num_classes).to(device)
    y_hat_nm = torch.mul(1-y_onehot, y_hat)
    y_hat_nm = y_hat_nm / torch.sum(y_hat_nm,dim=1, keepdim=True)
    y_hat = y_hat_nm*(1-yhat_max[:,None])+y_onehot*yhat_max[:,None]
    # Exectation of accuracy
    y_hat_np = y_hat.cpu().numpy()
    for i,row in enumerate(y_hat_np):
        if eacc < np.random.rand():
            # TODO: randperm is overkill. Just swap GT dim with a random dim.
            #y_hat_np[i,:] = row[np.random.permutation(num_classes)]
            j = myrandint(num_classes,wo=[y[i]])
            # SWAP
            tmp = y_hat_np[i,y[i]]
            y_hat_np[i,y[i]] = y_hat_np[i,j]
            y_hat_np[i,j] = tmp

    y_hat = torch.from_numpy(y_hat_np).to(device)
    return y, y_hat

def calculate_accuracy(y,y_hat,device):
    # calculated ground truth accuracy
    _, predicted = torch.max(y_hat, 1)
    correct = (predicted==y).sum()
    return torch.Tensor([float(correct) / y_hat.shape[0]]).to(device)

def calculate_accuracy_with_threshold(y,y_hat,device,th=None):
    if th is None:
        return calculate_accuracy(y,y_hat,device)
    # calculated ground truth accuracy
    val, predicted = torch.max(y_hat, 1)
    correct = ((predicted==y) & (val > th)).sum().float()
    return torch.Tensor([[float(correct) / y_hat.shape[0]]]).to(device)

def eval_accuracy(model,dataloader,device):
    with torch.no_grad():
        correct = 0.
        total = 0.
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted==y).sum().cpu().numpy()
            total += len(y)
        return correct / total

def accuracy_batch(Y_onehot: torch.tensor,Yhat_onehot: torch.tensor):
    _, y = torch.max(Y_onehot, -1)
    _, predicted = torch.max(Yhat_onehot, -1)
    match = (y ==predicted)
    return match.sum(dim=-1)/float(match.shape[-1])

def accuracy_batch_c(c: torch.tensor):
    # Have the same number of axes as input
    y_onehot,yhat_onehot = torch.split(c,[int(c.shape[-1]/2)]*2,dim=-1)
    return accuracy_batch(y_onehot,yhat_onehot)

def idx2onehot(y,num_classes,device):
    return nn.functional.one_hot(y, num_classes=num_classes).float().to(device)

def fold_batch(X,szBatch):
    # 2D matrix [szBatch*n X dim] to 3D matrix [szBatch X n X dim]
    return X.reshape(torch.Size([szBatch,int(X.shape[0]/szBatch)])+X.shape[1:])

def freeze_params(model):
    for par in model.parameters():
        par.requires_grad = False

def unfreeze_params(model):
    for par in model.parameters():
        par.requires_grad= True

def ratio2nsamples(ratio,N):
    ratio = np.array(ratio, dtype=float)
    ratio /= np.sum(ratio)
    nsamples =[]
    for r in ratio[:-1]:
        nsamples.append(int(r*N))
    nsamples.append(N-sum(nsamples))
    return nsamples


def clone_parameters(param_list):
    return [p.clone() for p in param_list]

def xplusag(xs, alpha,gs,no_grad=True):
    # xs = f.parameters()
    # alpha = -0.0001 for minimization
    # gs = grad(loss, params)
    # no_grad = True
    if no_grad:
        with torch.no_grad():
            _xplusag(xs, alpha, gs)
    else:
        _xplusag(xs, alpha,gs)

def _xplusag(xs, alpha, gs):
    for j,param in enumerate(xs):
        assert(param.data.shape == gs[j].shape)
        param.data = param.data+alpha*gs[j]

def mynorm(params):
    return sum([torch.norm(param.reshape(-1)) for param in params])

def get_device_from_model(model):
    return next(model.parameters()).device

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

from datetime import datetime
def now2str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

