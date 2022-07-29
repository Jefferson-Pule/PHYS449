'Neural Network Differential equations'
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
    def __init__(self, inputs, n_hidden, ntests , time_step, steps, df_x, df_y):
        super(Net, self).__init__()

        self.rnn= nn.RNN(input_size=2,hidden_size =n_hidden, num_layers=1, batch_first=True)
        self.sg= nn.Sigmoid()
        self.fc= nn.Linear(n_hidden, 2)
        self.df_x= df_x
        self.df_y= df_y
        self.ntests=ntests
        self.n_hidden=n_hidden
        self.steps=steps
        self.time_step=time_step

    # Feedforward function
    def forward(self, inputs):

        # Initiate values
        inputs=torch.tensor(inputs).float()
        hidden=torch.zeros(1,self.ntests,self.n_hidden)
 
        # Initiate List to collect different iterations
        xy=inputs   # Positions
        v=torch.zeros_like(xy).float() # velocities
        
        # Run single cell n_step times

        for step in range(self.steps):

            out, hidden=self.rnn(inputs, hidden)     #Update hidden tensor
            
            #Calculate velocities and positions and update RNN inputs
            hid=self.sg(hidden)
            velocity=self.fc(hid)
            vsize=velocity.size()
            velocity=torch.reshape(velocity, shape=[vsize[1],vsize[0],vsize[2]])
            
            inputs=inputs+self.time_step*velocity # New position
            
            #Save data
            v=torch.cat([v,velocity], axis=1)
            xy=torch.cat([xy,inputs], axis=1)

#        print("v",v)
#        print("xy",xy)

        return v, xy

    def loss(self,v,xy):
        
        #Calculate expected velocity at each position
        with torch.no_grad():
            v_rnn=torch.zeros_like(v).float()
        
            xy=xy.detach().numpy()

            for n_test in range(self.ntests):
                for step in range(self.steps):

                    x=xy[n_test][step][0]
                    y=xy[n_test][step][1]

                    v_rnn[n_test][step+1][0]=self.df_x(x,y)
                    v_rnn[n_test][step+1][1]=self.df_y(x,y)
        
#        print("v_rnn",v_rnn)
        
        # Calculate mean difference
        loss_diff=torch.mean(v_rnn-v)

        return loss_diff

    # Backpropagation function
    def backprop(self, inputs, optimizer):
        self.train()
        # Run the forward step and calculate the loss function
        v, xy = self.forward(inputs)
        obj_val=self.loss(v,xy)
        
        # Backpropagate        
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()     





