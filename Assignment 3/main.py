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

from NN import Net, train, evaluate



if __name__ == '__main__':
    #Command line arguments
    parser= argparse.ArgumentParser(description='Inputs')
    
    parser.add_argument("-json", default="data/arguments.json", help="input path to json file, default: data/arguments.json")
    parser.add_argument("-v", default=1 ,help="verbosity, default:1")
    parser.add_argument("-results", default="results" ,help="path to folder to save results, default: results")
    parser.add_argument("-xfield",help="expression of the x-component of the vector field")
    parser.add_argument("-yfield",help="expression of the y-component of the vector field")
    parser.add_argument("-lb",help="lower bound for initial conditions")
    parser.add_argument("-ub",help="upper bound for initial conditions")
    parser.add_argument("-ntests",help="number of test trajectories to plot")
 
    args = parser.parse_args()

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
    division=json_file['time division']

    #Initial conditions
    Xi=np.random.uniform(lb,up,size=(ntests))
    
    Yi=np.random.uniform(lb,up,size=(ntests))
    
    # Define global functions
    u="u"
    v="v"
    exec("{} = lambda x,y: {}".format(u,dx))
    exec("{} = lambda x,y: {}".format(v,dy))

    df_x= lambda x,y: u(x,y)
    df_y= lambda x,y: v(x,y)

    #Creat an instance of the model


    model=Net(n_hidden,df_x, df_y)
    optimizer = optim.Adam(model.parameters(), lr)
    train(epochs, 100, model, optimizer)

    
    def plot(k,xi,yi,u,v, model,t,division):
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

