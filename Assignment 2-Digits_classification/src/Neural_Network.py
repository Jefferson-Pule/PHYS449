'Neural Network to classify digits'
import numpy as np
import torch 
import torch.nn as nn #layer types
import torch.nn.functional as func #useful ML functions that are applied to the layers
import torch.optim as optim

class Net(nn.Module):
    '''
    Neural network class.
    Architecture:
        Two fully-connected layers fc1 and fc2.
        Two nonlinear activation functions sigmoid and softmax.
    '''
    def __init__(self, n_bits, n_hidden):
        super(Net, self).__init__()

        self.fc1= nn.Linear(n_bits, n_hidden)
        self.fc2= nn.Linear(n_hidden, 5) 
        
        
    # Feedforward function
    def forward(self, inputs):
        h = func.sigmoid(self.fc1(inputs))
        y = func.softmax(self.fc2(h), dim=1)
        return y
    
    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        
    # Backpropagation function
    def backprop(self, inputs, targets, loss, epoch, optimizer):
        self.train()
        
        # Transform targets to tensor form
        targets= torch.flatten(torch.from_numpy(targets))
        
        # Run our forward step once
        results=self.forward(inputs)
        
        # Calculate the accuracy of our results
        acc=self.accuracy(results,targets)
        
        # Back propagate the error
        obj_val= loss(results, (targets/2).type(torch.LongTensor))
        optimizer.zero_grad()
        obj_val.backward()         # Update the values 
        optimizer.step()

        return obj_val.item(), acc
    
    # Test function. Avoids calculation of gradients.
    def test(self, inputs, targets, loss, epoch):
        self.eval()

        # Run the NN without calculating gradients
        with torch.no_grad():
            targets= torch.flatten(torch.from_numpy(targets))
            results=self.forward(inputs)
            acc=self.accuracy(results,targets)
            cross_val= loss(results, (targets/2).type(torch.LongTensor))
        
            return cross_val.item(), acc

    # Calculates the porcentage of values that were correctly labeled by our NN    
    def accuracy(self, results, targets):

        prediction=2*torch.argmax(results, dim=1)
        accuracy_vect=torch.where(prediction-targets==0, 1,0)
        accuracy=torch.mean(accuracy_vect.to(torch.float32))
        

        return accuracy

    # Splits the data into inputs and targets for the training and also generates a test set.
    def getdata(self,data):

        #Dimensions
        dim=data.shape
        n_s=dim[0]       #number of samples
        i_d=dim[1]-1     #dimention of inputs
        
        np.random.shuffle(data)
        test_data, train_data=np.split(data,[3000])
        
        # Separate inputs and targets for test
        test_inputs=test_data[:,0:i_d]
        test_targets=test_data[:,[-1]]

        #Separate inputs and targets forTraining
        inputs=train_data[:,0:i_d]
        targets=train_data[:,[-1]]
        return inputs, targets, test_inputs, test_targets

