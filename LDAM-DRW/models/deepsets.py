import torch
import torch.nn as nn

### META MODEL ###
class InvariantModel(nn.Module):
    # rho sum_i phi(x_i)
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho
        self.relu = nn.ReLU()
    def forward(self, x):
        # compute the representation for each data point
        x = self.phi(x)
        #phi_out = x
        # Fix for the mini-batch case
        x = torch.mean(x, dim=-2, keepdim=True)    # Double Check!!!
        x = self.relu(x)
        # compute the output
        out = self.rho(x)

        return out

class ObjectPhi(nn.Module):
    def __init__(self, input_size=2, hidden_size=20, output_size=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(self.input_size, self.hidden_size)
        self.hidden2 = nn.Linear(self.hidden_size, self.hidden_size)

        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
   #     x = self.relu(x)
        return x

class ObjectRho(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Inputs to hidden layer linear transformation
        self.output = nn.Linear(self.input_size, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.output(x)
        x = torch.sigmoid(x)
        return x

# WORKING ON IT!!!
import pdb

class InvariantModel2(nn.Module):
    # DeepSets with a dimension-wise function
    def __init__(self, phidim: nn.Module, rho: nn.Module, num_classes: int):
        super().__init__()
        self.phidim = phidim
        self.rho = rho
        self.relu = nn.LeakyReLU()
        self.u = torch.empty((num_classes, 1),requires_grad=True) # Before sigmoid
        nn.init.xavier_normal_(self.u) # init
        self.softmax = nn.Softmax(dim=0)
        self.device = get_device_from_model(phidim)


    def forward(self, x):
#        pdb.set_trace()
        x1, x2 = x.split(int(x.shape[-1]/2), dim=-1)
        x = torch.stack((x1, x2), dim=-1)  # Torch Stack!

        x = self.phidim(x)
        x = torch.squeeze(x)
#        w = self.softmax(self.u).to(self.device)
#        x = torch.matmul(x, w)

#        pdb.set_trace()
        x = self.relu(x)
        x = torch.mean(x, dim=[-2,-1])
        if self.rho is None:
            return x
        # compute the output
        else:
            out = self.rho(x)
            return out

# train network
class FCNetwork(nn.Module):
    def __init__(self, input_size=2, num_classes=3):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(self.input_size, 40)
        self.hidden2 = nn.Linear(40, 40)
        self.hidden3 = nn.Linear(40, 40)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(40, self.num_classes)
        
        # Define sigmoid activation and softmax output 
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.output(x)
       
        return x

class FCNetworkSoftmax(FCNetwork):
    def __init__(self, input_size=2, num_classes=3):
        super().__init__(input_size=input_size, num_classes=num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        out = super(FCNetworkSoftmax,self).forward(x)
        return self.softmax(out)

# To remove the dependency on tools.py, the following function is copied here.
def get_device_from_model(model):
    return next(model.parameters()).device
