
#Inhehrite Matrix Operations
import numpy as np
import torch 
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable 


# Fully Visible RBM Structure 
class RBM(nn.Module):
    def __init__(self, n_vis, k):
        
        super(RBM, self).__init__()
        
        self.N=n_vis                    # Number of Visible Nodes (size of our system)
        self.J = torch.rand(n_vis)      # Initial guess for the 
        self.k = k
        
        self.buildlattice()  
        
    # Gets the visible samples, and the Metropolis samples
    def forward(self,v):

        vn=self.Metropolis_Hastings(v)
        
        return 2*v-1, vn
    
    def train(self, epochs, data, learning_rate, verbosity):

        data=torch.tensor(data, dtype=torch.float32)
        lr=learning_rate
        for epoch in range(epochs):
            v,vn=self.forward(data)

            grad=self.gradient(v,vn)
            
            self.J+=lr*grad     #Update the value of J based on the gradient
            
            loss=self.KL_divergence(v)[0]

            if (epoch+1) % verbosity == 0:
                print('Epoch [{}/{}]'.format(epoch+1, epochs)+\
                          '\tTraining Loss: {:.4f}'.format(loss))

        print('Final training loss: {:.4f}'.format(loss))
        return self.J, loss

    def gradient(self,v,vn):
        
        positive_phase=self.get_pairs(v)
        negative_phase=self.get_pairs(vn)
        
        return positive_phase-negative_phase
    
    # Gets the pair wise convination xiyi and saves it in an array
    def get_pairs(self,v):
        XY=[]      
        
        for n in range(len(self.nearest_neighbor)):
            xnyn=v[:,self.nearest_neighbor[n][0]]*v[:,self.nearest_neighbor[n][1]]
            XY.append(torch.mean(xnyn))
            
        return torch.tensor(XY, dtype=torch.float32)


    def energy(self,v):
        
        eloc=torch.zeros(size=[v.shape[0]])
        
        for n in range(len(self.nearest_neighbor)):
            eloc-=self.J[self.nearest_neighbor[n][0]]*v[:,self.nearest_neighbor[n][0]]*v[:,self.nearest_neighbor[n][1]]
        
        return eloc
    
    def sample_from_p(self,prob,vn, vholder):
        
        accept_change=torch.reshape(torch.bernoulli(prob), shape=[vn.shape[0],1]) # return 1 if change is accepted or 0 if not based on probabilities
        accept_change=accept_change*torch.ones(size=vn.shape)          # Generates a tensor of the size of vn that will update based on previous step
        new_sample=torch.where(accept_change==1, vholder, vn)          # Choose from vholder if change is accepted or from vn if change is not accepted

        return new_sample

    def Metropolis_Hastings(self, v):
        vn=torch.add(torch.mul(v,2),-1)     #vn vissible layer
        
        for n in range(self.k):

            change=2*torch.randint(0,2, size=v.shape, dtype=torch.float32)-1     #Random change
            vholder=change*vn                                                    #Possible change
            neg_energy_diff=-(self.energy(vn)-self.energy(vholder))              #Calculate energy difference
            prob=torch.exp(-F.relu(neg_energy_diff))                             #Calculate Prob
            vn= self.sample_from_p(prob,vn,vholder)                              #Sample from vn or vholder based on probability 
            
        return vn

    # Builds an array that contains the nearest neighbors index
   
    def buildlattice(self):
        self.nearest_neighbor=[]
        for spin in range(self.N):
            self.nearest_neighbor.append([spin, (spin+1)%self.N])

    def KL_divergence(self,p):
        #Calculate the unique samples and their frequencies 

        unique_p, counts_p= torch.unique(p, dim=0, return_counts=True)
        prob_p=1/(torch.sum(counts_p))*counts_p                    #Probability of each unique sample

        logp=torch.log(prob_p)
        neg_entropy=0                      #Negative Entropy
        pxenergy=0                         #First term of the cross entropy
        partition=0                        #Partition function 
        
        for x in range(len(unique_p)):
            neg_entropy+=(prob_p[x]*logp[x]).numpy()
            eloc=self.energy(torch.reshape(unique_p[x], shape=(1,unique_p.shape[1])))
            pxenergy+=prob_p[x].numpy()*eloc.numpy()
            partition+=np.exp(-eloc.numpy())
            
        KL=neg_entropy+pxenergy+np.log(partition) #KL divergence

        return KL

