
#Inhehrite Matrix Operations
import numpy as np
import torch 
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable 
import matplotlib.pyplot as plt
import os 

# Fully Visible RBM Structure 
class RBM(nn.Module):
    def __init__(self, n_vis, k):
        
        super(RBM, self).__init__()
        
        self.N=n_vis                    # Number of Visible Nodes (size of our system)
        self.J = torch.rand(n_vis)      # Initial guess for the 
        self.k = k
        
        self.buildlattice()    
        
    def forward(self,v):

        vn=self.Metropolis_Hastings(v)
        
        return 2*v-1, vn
    
    def gradient(self,v,vn):
        
        positive_phase=self.get_pairs(v)
        negative_phase=self.get_pairs(vn)
        
        return positive_phase-negative_phase
    
    def train(self, epochs, data, learning_rate, verbosity):
        data=torch.tensor(data, dtype=torch.float32)
        lr=learning_rate
        for epoch in range(epochs):
            v,vn=self.forward(data)

            grad=self.gradient(v,vn)
            
            self.J+=lr*grad

            if (epoch+1) % verbosity == 0:
                print('Epoch [{}/{}]'.format(epoch+1, epochs)+\
                          '\tTraining Loss: {:.4f}'.format(torch.mean(grad)))

        print('Final training loss: {:.4f}'.format(torch.mean(grad)))
        return self.J, grad
    
    def get_pairs(self,v):
        XY=[]
        
        for n in range(len(self.nearest_neighbor)):
            xnyn=v[:,self.nearest_neighbor[n][0]]*v[:,self.nearest_neighbor[n][1]]
            XY.append(torch.mean(xnyn))
            
        return torch.tensor(XY, dtype=torch.float32)

    def buildlattice(self):
        self.nearest_neighbor=[]
        for spin in range(self.N):
            self.nearest_neighbor.append([spin, (spin+1)%self.N])

    def energy(self,v):
        
        eloc=torch.zeros(size=[v.shape[0]])
        
        for n in range(len(self.nearest_neighbor)):
            eloc-=self.J[self.nearest_neighbor[n][0]]*v[:,self.nearest_neighbor[n][0]]*v[:,self.nearest_neighbor[n][1]]
        
        return eloc
    
    def sample_from_p(self,prob,vn, vholder):
        
        accept_change=torch.reshape(torch.bernoulli(prob), shape=[vn.shape[0],1])
        mean_accept_change=torch.mean(accept_change)
        accept_change=accept_change*torch.ones(size=vn.shape)
        new_sample=torch.where(accept_change==1, vholder, vn)

        return new_sample

    def Metropolis_Hastings(self, v):
        vn=torch.add(torch.mul(v,2),-1)
        
        for n in range(self.k):

            change=2*torch.randint(0,2, size=v.shape, dtype=torch.float32)-1
            vholder=change*vn
            neg_energy_diff=-(self.energy(vn)-self.energy(vholder))
            prob=torch.exp(-F.relu(neg_energy_diff))
            vn= self.sample_from_p(prob,vn,vholder)
            
        return vn




