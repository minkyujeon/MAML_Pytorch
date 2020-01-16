import torch
from torch import nn
from torch import optim
from torch import autograd
import numpy as np

class Learner(nn.Module):
    # 매 episode마다 theta와 meta-train set을 이용하여 theta'을 update
    # 매 episode마다 theta'과 meta-test set을 이용하여 theta를 update
    
    def __init__(self, net, alpha, *args):
        super(Learner, self).__init__()
        self.alpha = alpha

        self.net_theta = net(*args) # theta : prior / general
        self.net_theta_prime = net(*args) # theta' : task specific
        self.optimizer = optim.SGD(self.net_theta_prime.parameters(), self.alpha)

    def forward(self, support_x, support_y, query_x, query_y, num_updates):
        # 현재 theta로부터 theta'를 구하기 위함 (fine tune)
        # specific task (theta')
        
        for theta, theta_prime in zip(self.net_theta.modules(), self.net_theta_prime.modules()):
            if isinstance(theta_prime, nn.Linear) or isinstance(theta_prime, nn.Conv2d) or isinstance(theta_prime, nn.BatchNorm2d):
                theta_prime.weight.data = theta.weight.data.clone()
                if theta_prime.bias is not None:
                    theta_prime.bias.data = theta.bias.data.clone()
                    # clone():copy the data to another memory but it has no interfere with gradient back propagation
        
        # print('support_x:',support_x.shape) # [5, 1, 28, 28]
        for i in range(num_updates):
            loss, pred = self.net_theta_prime(support_x, support_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # meta gradient 계산
        # meta learner로 theta를 update 하기위한 theta' net의 gradient를 구함
        loss, pred = self.net_theta_prime(query_x, query_y)
        # pred : [dataset_size, n_way] (5,5)
        _, indices = torch.max(pred, dim=1)
        correct = torch.eq(indices, query_y).sum().item()
        acc = correct/query_y.size(0)
        
        # create_grad=True로 하면 autograd.grad 후에도 backward를 다시 호출 가능(Hessian을 위함)
        gradient_pi = autograd.grad(loss, self.net_theta_prime.parameters(), create_graph=True) #create_graph : for second derivative
        
        return loss, gradient_pi, acc
    
    def net_forward(self, support_x, support_y):
        # theta update (general)
        # metalearner에서 merged gradients를 net_theta network에 wirte하기 위함
        loss, pred = self.net_theta(support_x, support_y)
        return loss, pred
        
        
class MetaLearner(nn.Module):
    # net_theta' network의 다양한 task에 대한 loss를 받아서 모든걸 합한 general한 initialize parameter를 찾음
    # 매 episode마다 theta'과 meta-test set을 이용하여 theta를 update

    def __init__(self, net, net_args, n_way, k_shot, meta_batch_size, alpha, beta, num_updates):
        super(MetaLearner, self).__init__()
        
        self.n_way = n_way
        self.k_shot = k_shot
        self.meta_batch_size = meta_batch_size
        self.beta = beta
        self.num_updates = num_updates
        
        self.learner = Learner(net, alpha, *net_args)
        self.optimizer = optim.Adam(self.learner.parameters(), lr=beta)
    
    def meta_update(self, dummy_loss, sum_grads_pi):
        # sum_gradients를 활용하여 theta_parameter를 update함

        hooks = []
        for k, v in enumerate(self.learner.parameters()):
            def closure():
                key = k
                return lambda grad: sum_grads_pi[key]
            
            hooks.append(v.register_hook(closure()))
            # register_hook : If you manipulate the gradients, the optimizer will use these new custom gradients to update the parameters
        
        self.optimizer.zero_grad()
        dummy_loss.backward() # dummy_loss : summed gradients_pi (general한 theta network을 위해)
        self.optimizer.step()
        
        for h in hooks:
            h.remove()
        
    def forward(self, support_x, support_y, query_x, query_y):
        
        # 매 episode마다 Learner에 의해 학습됨 -> parameter theta'에 대한 loss들을 get
        # loss를 get하고 합쳐서 theta를 update
        
        sum_grads_pi = None
        meta_batch_size = support_y.size(0) # 5
        
        accs = []
        for i in range(meta_batch_size):
            _, grad_pi, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
            accs.append(episode_acc)
            if sum_grads_pi is None:
                sum_grads_pi = grad_pi
            else:
                sum_grads_pi = [torch.add(i,j) for i,j in zip(sum_grads_pi, grad_pi)] #모든 task에 대해 grad_pi를 sum하여 theta를 구할 때 반영
        dummy_loss, _ = self.learner.net_forward(support_x[0], support_y[0])
        # print('support_x[0]:',support_x[0].shape) #[5, 1, 28, 28]
        self.meta_update(dummy_loss, sum_grads_pi)
        
        return accs
    
    def pred(self, support_x, support_y, query_x, query_y):
        meta_batch_size = support_y.size(0)
        accs = []
        
        for i in range(meta_batch_size):
            _, _, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
            accs.append(episode_acc)
            
        return np.array(accs).mean()