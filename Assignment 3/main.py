# Digits Classification
import numpy as np
import  json, argparse, sys, os

import torch 
import torch.nn as nn #layer types
import torch.nn.functional as func #useful ML functions that are applied to the layers
import torch.optim as optim

import matplotlib.pyplot as plt

sys.path.append('src')
from Neural_Network import Net

def run_demo(lr,epochs, model, xi,yi, time_step):
    for param in model.parameters():
        param.requires_grad = True
    # Divide data into inputs and lables, but also into a 3000 test 
    X, Y, T,  XYini = model.getdata(xi,yi, time_step) 

    # Define an optimizer and the loss function
    
#    optimizer = optim.SGD(model.parameters(), lr)  # We use gradient descent 

    # If you want you could try Adam
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.99)

    # The loss function is the cross entropy.
    loss= torch.nn.MSELoss(reduction= 'mean')     

    #Lists that will contain the information for each epoch
    loss_train= []
    acc_train=[]
#    loss_test= []
#    acc_test=[]
    num_epochs= int(epochs)

    # Training loop
    for epoch in range(1, num_epochs + 1):
  
        train_val, accuracy= model.backprop(X, Y , T , XYini, loss, epoch, optimizer)
        loss_train.append(train_val)
        acc_train.append(accuracy)        
            
#        test_val, test_acc= model.test(test_inputs, test_targets, loss, epoch)       
#        loss_test.append(test_val)
#        acc_test.append(test_acc)

        if (epoch+1) % 2 == 0:
            print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                      '\tTraining Loss: {:.4f}'.format(train_val)+\
                      '\tTraining Accuracy: {:.4f}'.format(accuracy))

    print('Final training loss: {:.4f}'.format(loss_train[-1]))
    print('Final training accuracy: {:.4f}'.format(acc_train[-1]))

    return  loss_train, acc_train

#def plot_results(training, test, type_of_graph):
#    assert len(training)==len(test), 'Length mismatch between the curves'
#    num_epochs= len(training)

    # Plot saved in results folder
#    plt.plot(range(num_epochs), training, label= "Training "+type_of_graph, color="blue")
#    plt.plot(range(num_epochs), test, label= "Test "+type_of_graph, color= "green")
#    plt.legend()
#    plt.savefig(args.output + '/fig'+type_of_graph+'.pdf')
#    plt.close()


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
    dx=args.xfield
    dy=args.xfield

    f=open(args.json)
    json_file=json.load(f)
    
    lr=json_file['learning rate']   
    epochs=json_file['Epochs']    
    n_hidden=json_file['n_hidden']  #number of hidden
    time_step=json_file['time step']

    #Initial conditions
    xi=np.random.uniform(lb,up)
    yi=np.random.uniform(lb,up)

    # Define global functions
    u="u"
    v="v"
    exec("{} = lambda x,y: {}".format(u,dx))
    exec("{} = lambda x,y: {}".format(v,dy))

    df_x= lambda x,y: u(x,y)
    df_y= lambda x,y: v(x,y)

    #Creat an instance of the model

    model=Net(n_hidden,time_step, xi, yi, df_x, df_y)
    
    #  Run the model 
    loss_train, acc_train=run_demo(lr,epochs, model, xi,yi, time_step)
    model.reset()

    #Plot and save
#    plot_results(loss_train, loss_test, "loss")
#    plot_results(acc_train, acc_test, "accuracy")