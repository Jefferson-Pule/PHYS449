'Neural Network to classify digits'
import numpy as np
import torch 
import torch.nn as nn #layer types
import torch.nn.functional as func #useful ML functions that are applied to the layers
import torch.optim as optim
import matplotlib.pyplot as plt

class Net(nn.Module):
    '''
    VAE.
    Architecture:
        Encoder: 2 convolutional layers conv1 and conv2 and 2 linear layers  fc1, fc2.
        Sampling: Use of parametrization trick to get data for the decoder
        Decoder: 1 linear layers and 2 convolutional layers inverting the dimensions of the Encoder.
    '''
    def __init__(self, n_bits, lattent_dims):
        super(Net, self).__init__()
        
        # Encoder layers

        self.en_conv1_layer=nn.Sequential(\
            nn.Conv2d(in_channels=1,out_channels=5,kernel_size=3,stride=1,padding=1),\
            nn.ReLU())
        
        self.en_conv2_layer=nn.Sequential(\
            nn.Conv2d(in_channels=5,out_channels=10,kernel_size=4,stride=2,padding=1),\
            nn.ReLU())
        
        self.en_fc1= nn.Linear(7*7*10,lattent_dims)
        
        self.en_fc2= nn.Linear(7*7*10, lattent_dims) 
        
        # Decoder layers
        self.de_fc= nn.Linear(lattent_dims, 7*7*10)
        
        self.de_covTranspose2_layer=nn.Sequential(\
            nn.ConvTranspose2d(in_channels=10, out_channels=5, kernel_size=4, stride=2, padding=1),\
            nn.ReLU())
        
        self.de_covTranspose1_layer=nn.Sequential(\
            nn.ConvTranspose2d(in_channels=5, out_channels=1, kernel_size=3, stride=1, padding=1),\
            nn.ReLU())
        

    def encoder(self, inputs):
        
        # Convolutional Layer 
        cv_1=self.en_conv1_layer(inputs)     
        cv_2=self.en_conv2_layer(cv_1)
        flatten=cv_2.reshape(cv_2.size(0),-1)
        
        # calculate mean and log of standar deviation
        mean=self.en_fc1(flatten)                 
        log_sigma=self.en_fc2(flatten)


        return mean, log_sigma

    def sampling(self, mean, log_sigma):
        
        # sample from a N(0,1) and then find the sample by using properties of Normal distribution

        epsilon=torch.randn(size=log_sigma.shape) 
        sample=mean+torch.exp(log_sigma)*epsilon  # Sample is mean +sigma*epsilon
        
        return sample
    
    def decoder(self, sample, ns):

        # Get back to the original output size
        inv_linear=self.de_fc(sample)
        unflatten=inv_linear.reshape([ns, 10, 7, 7])
        cvT_2=self.de_covTranspose2_layer(unflatten)
        cvT_1=self.de_covTranspose1_layer(cvT_2)
                
        return cvT_1
    
    def forward(self, inputs, ns):
        
        # Apply encoding, parametrization trick (sampling), and decoder

        mean, log_sigma= self.encoder(inputs)      
        sample=self.sampling(mean, log_sigma)
        output=self.decoder(sample,ns)
        
        return mean, log_sigma, output
    
    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        
    # Backpropagation function
    def backprop(self, inputs, loss, epoch, optimizer, ns):
        self.train()
        # Run the VAE and calculate the loss
        mean, log_sigma, outputs = self.forward(inputs, ns)
        obj_val= loss(inputs,mean, log_sigma, outputs, ns)          
        
        # Backpropagate        
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()    



    # Splits the data into inputs and targets for the training and also generates a test set.
    def getdata(self,data):

        #Dimensions
        dim=data.shape
        n_s=dim[0]       #number of samples
        i_d=dim[1]-1     #dimention of inputs
        
        train_data=data
        
        #Separate inputs from targets for Training 
        inputs=train_data[:,0:i_d]
        inputs=inputs/255                       # Normalization of the data
        inputs=np.reshape(inputs, newshape=(n_s, 1, int(np.sqrt(i_d)), int(np.sqrt(i_d))))

        return inputs
