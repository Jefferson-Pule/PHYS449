# Digits Classification
import numpy as np
import  json, argparse, sys, os

import torch 
import torch.nn as nn #layer types
import torch.nn.functional as func #useful ML functions that are applied to the layers
import torch.optim as optim

import matplotlib.pyplot as plt


sys.path.append('src')

#from Neural_Network import Net

from Neural_Network import Net

def run_demo(lr, epochs, model, inputs):

    #Choose an optimazer
    optimizer = optim.SGD(model.parameters(), lr)  # We use gradient descent 
   
    #Lists that will contain the information for each epoch
    loss_train= []
    num_epochs= int(epochs)

    # Training loop
    for epoch in range(1, num_epochs + 1):
  
        train_val = model.backprop(inputs, optimizer)
        loss_train.append(train_val)
        if verb:
            if (epoch+1) % verb_epoch == 0:
                print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                          '\tTraining Loss: {:.4f}'.format(train_val))

    print('Final training loss: {:.4f}'.format(loss_train[-1]))

    return  loss_train


if __name__ == '__main__':
    #Command line arguments
    parser= argparse.ArgumentParser(description='Inputs')
    
    parser.add_argument("-json", default="data/arguments.json", help="input path to json file, default: data/arguments.json")
    parser.add_argument("-results", default="results" ,help="path to folder to save results, default: results")
    parser.add_argument("-xfield",help="expression of the x-component of the vector field")
    parser.add_argument("-yfield",help="expression of the y-component of the vector field")
    parser.add_argument("-lb",help="lower bound for initial conditions")
    parser.add_argument("-ub",help="upper bound for initial conditions")
    parser.add_argument("-ntests",help="number of test trajectories to plot")
    parser.add_argument("-verbose", default="True" ,help="Verbose mode True or False. default True")
    parser.add_argument("-verbose_epoch", default="2" ,help="After how many epochs to give the report, default 2")
 
    args = parser.parse_args()
    
    # Verbose
    verb=bool(args.verbose)
    verb_epoch=int(args.verbose_epoch)

    #Load data
    lb=float(args.lb)
    up=float(args.ub)
    ntests=int(args.ntests)
    dx=args.xfield
    dy=args.yfield

    f=open(args.json)
    json_file=json.load(f)
    
    lr=json_file['learning rate']   
    epochs=json_file['Epochs']    
    n_hidden=json_file['n_hidden']  #number of hidden
    t=json_file['final time']
    steps=json_file['time division']
    time_step=t/steps

    #Initial conditions
    Xi=np.random.uniform(lb,up,size=(ntests,1))
    Yi=np.random.uniform(lb,up,size=(ntests,1))

    # Process data 
    X=np.reshape(np.concatenate((Xi,Yi), axis=1),(ntests,1,2)) # Our input


    # Define global functions
    u="u"
    w="w"
    exec("{} = lambda x,y: {}".format(u,dx))
    exec("{} = lambda x,y: {}".format(w,dy))

    df_x= lambda x,y: u(x,y)
    print(df_x(1,1))
    df_y= lambda x,y: w(x,y)

    #Creat an instance of the model

    model=Net(X, n_hidden, ntests , time_step, steps, df_x, df_y)
    #  Run the model 
    loss_train=run_demo(lr, epochs, model, X)
    
    def plot(k,xi,yi,u,w, model,t,division):
        output=evaluate(xi,yi, model, t, division)
        output=torch.reshape(output, (output.shape[0],output.shape[2]))
        x,y=np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
        dx=u(x,y)
        dy=v(x,y)
        if k==0:
            plt.quiver(x,y,dx,dy)
        plt.plot(xi,yi, 'ro')
        plt.plot(output[:,0],output[:,1])

    k=0
    for t in range(ntests):
        xi=Xi[k]
        yi=Yi[k]
        plot(k,xi,yi,u,v,model,t,division)
        k+=1
    plt.savefig(args.results+'/fig1.pdf')
    plt.show()

