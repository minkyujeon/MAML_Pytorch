import torch
from torch import nn
from torch import optim
from torch import autograd
import numpy as np

class Learner(nn.Module):
    
    def __init__(self, net, alpha, *args):
        super(Learner, self).__init__()
        self.alpha = alpha

        self.net_theta = net(*args) # theta : prior / general
        self.net_phi = net(*args) # phi : task specific
        self.optimizer = optim.SGD(self.net_phi.parameters(), self.alpha) # Learner(inner loop, for task specific phi)

    def forward(self, support_x, support_y, query_x, query_y, num_updates):
        # 현재 theta로부터 phi를 구하기 위함 (fine tune)
        # theta를 phi에 copy

        with torch.no_grad():
            for theta, phi in zip(self.net_theta.modules(), self.net_phi.modules()):
                if isinstance(phi, nn.Linear) or isinstance(phi, nn.Conv2d) or isinstance(phi, nn.BatchNorm2d):
                    phi.weight.data = theta.weight.clone() # 이 clone()을 안하면 성능이 매우 떨어짐(학습이 안됨)
                    if phi.bias is not None:
                        phi.bias.data = theta.bias.clone()
                        # clone():copy the data to another memory but it has no interfere with gradient back propagation (cf. deepcopy)

        # support_x: [5, 1, 28, 28]
        for i in range(num_updates):
            loss, pred = self.net_phi(support_x, support_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # meta gradient 계산
        # meta learner로 theta를 update 하기위한 phi net의 gradient를 구함
        loss, pred = self.net_phi(query_x, query_y)
        # pred : [dataset_size, n_way] (5,5)
        _, indices = torch.max(pred, dim=1)
        correct = torch.eq(indices, query_y).sum().item()
        acc = correct/query_y.size(0)

        # create_grad=True로 하면 autograd.grad 후에도 backward를 다시 호출 가능(Hessian을 위함)
        gradient_phi = autograd.grad(loss, self.net_phi.parameters(), create_graph=True) #create_graph : for second derivative

        return loss, gradient_phi, acc

    def net_forward(self, support_x, support_y):
        # theta update (general)
        # metalearner에서 merged gradients를 net_theta network에 wirte하기 위함
        loss, pred = self.net_theta(support_x, support_y)
        return loss, pred

class MetaLearner(nn.Module):
    # net_phi network의 다양한 task에 대한 loss를 받아서 모든걸 합한 general한 initialize parameter를 찾음
    # 매 episode마다 phi와 meta-test set을 이용하여 theta를 update

    def __init__(self, net, net_args, n_way, k_shot, meta_batch_size, alpha, beta, num_updates):
        super(MetaLearner, self).__init__()

        self.n_way = n_way
        self.k_shot = k_shot
        self.meta_batch_size = meta_batch_size
        self.beta = beta
        self.num_updates = num_updates

        self.learner = Learner(net, alpha, *net_args)
        self.optimizer = optim.Adam(self.learner.parameters(), lr=beta)

    def meta_update(self, dummy_loss, sum_grads_phi):
        # sum_gradients를 활용하여 theta_parameter를 update
        hooks = []
        for k, v in enumerate(self.learner.parameters()):
            def closure():
                key = k
                return lambda grad: sum_grads_phi[key]

            hooks.append(v.register_hook(closure()))
            # register_hook : If you manipulate the gradients, the optimizer will use these new custom gradients to update the parameters
            # If you want to save gradients
            # The purpose of this piece of code is to investigate how to use modified gradient to update parameters.

        self.optimizer.zero_grad()
        dummy_loss.backward() # dummy_loss : summed gradients_phi (general한 theta network을 위해)
        self.optimizer.step()

        for h in hooks:
            h.remove()

    def forward(self, support_x, support_y, query_x, query_y):
        # 매 episode마다 Learner에 의해 학습됨 -> parameter theta'에 대한 loss들을 get
        # loss를 get하고 합쳐서 theta를 update

        sum_grads_phi = None
        meta_batch_size = support_y.size(0) # 5

        accs = []
        for i in range(meta_batch_size):
            _, grad_phi, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
            accs.append(episode_acc)
            if sum_grads_phi is None:
                sum_grads_phi = grad_phi
            else:
                sum_grads_phi = [torch.add(i,j) for i,j in zip(sum_grads_phi, grad_phi)] #모든 task에 대해 grad_pi를 sum하여 theta를 구할 때 반영
        dummy_loss, _ = self.learner.net_forward(support_x[0], support_y[0])
        # support_x[0] : [5, 1, 28, 28]
        self.meta_update(dummy_loss, sum_grads_phi)

        return accs

    def pred(self, support_x, support_y, query_x, query_y):
        meta_batch_size = support_y.size(0)
        accs = []
        
        for i in range(meta_batch_size):
            _, _, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
            accs.append(episode_acc)
            
        return np.array(accs).mean()
