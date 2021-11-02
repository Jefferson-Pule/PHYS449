import torch
import torch.nn as nn #layer types
import numpy as np
class Net(nn.Module):
    '''
    Neural network class.
    Architecture:
        Two fully-connected layers fc1 and fc2.
        Two nonlinear activation functions sigmoid and softmax.
    '''
    def __init__(self, n_hidden ,df_x, df_y):
        super(Net, self).__init__()

        self.fc1= nn.Linear(1, n_hidden)
        self.fc2= nn.Linear(n_hidden, n_hidden)
        self.fc3= nn.Linear(n_hidden, 2)
        self.df_x= df_x
        self.df_y= df_y

    # Feedforward function
    def forward(self,xiyi, t):
        h = torch.sigmoid(self.fc1(t))
        p = torch.tanh(self.fc2(h))
        r = self.fc3(p)

        return xiyi+t*r
      
    def loss(self, xiyi, t):
      t.requires_grad = True
      outputs = self.forward(xiyi,t)
      #Calculate the gradient of x in function of t
      grads_x = torch.autograd.grad(outputs[:,0], t, grad_outputs=torch.ones_like(outputs[:,0]),
                          create_graph=True, retain_graph=True)[0]
      
      #Calculate the gradient of y in function of 
      grads_y = torch.autograd.grad(outputs[:,1], t, grad_outputs=torch.ones_like(outputs[:,1]),
                          create_graph=True, retain_graph=True)[0]
      
      #The loss is the sum of the mean squares and the expected gradient.

      diff_x = (grads_x - torch.tensor(self.df_x(xiyi[:,0],xiyi[:,1]))) ** 2
      diff_y = (grads_y - torch.tensor(self.df_y(xiyi[:,0],xiyi[:,1]))) ** 2
      loss = (diff_x.sum() + diff_y.sum()) / diff_x.shape[0]
      return  loss
  


def train(num_epochs, num_iters, model, optimizer):
    loss_train= []
    acc_train=[]
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        for _ in range(num_iters):
            # get the inputs; data is a list of [inputs, labels]
            t=torch.rand(1)*5
            inputs = 2*torch.rand((100,2)) - 1
   
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = model.loss(inputs,t) 
            loss.backward()
            optimizer.step() 

            loss_train.append(loss)
            
        if (epoch+1) % 2 == 0:
            print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                      '\tTraining Loss: {:.4f}'.format(loss))
    print('Final training loss: {:.4f}'.format(loss_train[-1]))
    print('Finished Training')
 
def evaluate(xi,yi,model, t, division):
    xiyi=torch.tensor([[xi,yi]]).float()

    #points to evaluate
    
    point=[]
    for t in np.linspace(0,3,num=100):
        t=torch.tensor([t]).float()
        xtyt=model.forward(xiyi,t)
        point.append(xtyt.data)
    return torch.stack(point)




