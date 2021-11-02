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
    def __init__(self, n_hidden, time_step, xi, yi, df_x, df_y):
        super(Net, self).__init__()

        self.fc1= nn.Linear(4, n_hidden)
        self.fc2= nn.Linear(n_hidden, n_hidden)
        self.fc3= nn.Linear(n_hidden, 2)
        self.df_x= df_x
        self.df_y= df_y

        self.f_t= lambda t,x,y, xyini: torch.add(xyini,torch.multiply(t,self.forward(torch.cat((x,y,t), axis=1).float())))

    # Feedforward function
    def forward(self, inputs):
        h = torch.sigmoid(self.fc1(inputs))
        tanh = torch.nn.Tanh()
        p=tanh(self.fc2(h))
        y=self.fc3(p)
        return y
    
    # Reset function for the training weights
    # Use if the same network is trained multiple times.
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        
    # Backpropagation function
    def backprop(self, X, Y, T, XYini, loss, epoch, optimizer):
        self.train()
        # Find the resulting derivatives and the actual derivatives
        
        results, targets=self.derivatives(T, X, Y, XYini, self.f_t, self.df_x, self.df_y, optimizer)
        print("results", results)
        print("targets", targets)
        
        # Calculate the accuracy of our results
        acc=self.accuracy(results,targets)

        # Back propagate the error
        obj_val= loss(results, targets)

#        obj_val=torch.autograd.Variable(loss(results, targets),requires_grad=True)
        optimizer.zero_grad()
        obj_val.backward()         # Update the values 
        optimizer.step()

        return obj_val.item(), acc
    
    # Test function. Avoids calculation of gradients.
 #   def test(self, inputs, targets, loss, epoch):
 #       self.eval()

        # Run the NN without calculating gradients
 #       with torch.no_grad():
 #           targets= torch.flatten(torch.from_numpy(targets))
 #           results=self.forward(inputs)
 #           acc=self.accuracy(results,targets)
 #           cross_val= loss(results, (targets/2).type(torch.LongTensor))
        
 #           return cross_val.item(), acc

    # Calculates the porcentage of values that were correctly labeled by our NN    
    def accuracy(self, results, targets):

        accuracy=torch.mean(torch.add(results, targets, alpha=-1))

        return accuracy

    # Splits the data into inputs and targets for the training and also generates a test set.
    def getdata(self,xi,yi, time_step):

        t=np.arange(0,1, time_step)

        x=np.arange(-1,1,0.1)
        y=np.arange(-1,1,0.1)        
        x,y=np.meshgrid(y,x)
        xy=np.column_stack((y.ravel(),x.ravel()))

        inp=np.zeros((t.shape[0],xy.shape[0],4))
        k=0
        for i in t:
            zeros=np.zeros(xy.shape)+i
            inp[k]+=np.append(xy,zeros, axis=1)
            k+=1
        inp=np.reshape(inp, (inp.shape[0]*inp.shape[1],inp.shape[2]))

        X=inp[:,0]
        X=np.reshape(X,(X.shape[0],1))

        Y=inp[:,1]
        Y=np.reshape(Y,(Y.shape[0],1))

        T=inp[:,2:4]
        T=np.reshape(T,(T.shape[0],2))

        Xini=np.full((X.shape),xi)
        Yini=np.full((X.shape),yi)
        XYini=np.concatenate((Xini,Yini), axis=1)

        XYini=torch.tensor(XYini,requires_grad=True).float()
        X=torch.tensor(X, requires_grad=True).float()
        Y=torch.tensor(Y, requires_grad=True).float()
        T=torch.tensor(T, requires_grad=True).float()
        print(XYini,"XYini")
        print(X,"X")
        print(Y,"Y")
        print(T,"T")
        return X, Y, T, XYini

    
    # Calculate the exact derivative and the nn derivative in function of time
    def df(self ,x,y, df_x, df_y):
        dx=df_x(x,y)
        dy=df_y(x,y)
        return torch.cat((dx,dy), axis=1)

    def derivatives(self, T, X, Y, XYini, f_t, df_x, df_y, optimizer):       
        outputs=f_t(T,X,Y, XYini)
        out_x=torch.reshape(outputs[:,0].detach(),X.shape)
        out_y=torch.reshape(outputs[:,1].detach(),Y.shape)
#        outputs.backward(gradient=torch.ones(outputs.size()))
#        df_nn=T.grad
        print("outputs",outputs)
        df_nn=torch.autograd.grad(outputs, T, grad_outputs=torch.ones(outputs.size()), create_graph=True)
        print("df_nn",df_nn)
        df_exp=self.df(out_x,out_y, df_x, df_y)
        optimizer.zero_grad()
        return  df_nn[0], df_exp
        




