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

def run_demo(lr,epochs, model, data):

    # Divide data into inputs and lables, but also into a 3000 test 
    inputs, targets, test_inputs, test_targets= model.getdata(data)

    #Transform the numpy arrays into vectors
    inputs= torch.from_numpy(inputs)
    inputs=inputs.float()

    test_inputs=torch.from_numpy(test_inputs)
    test_inputs=test_inputs.float()

    # Define an optimizer and the loss function
    
    optimizer = optim.SGD(model.parameters(), lr)  # We use gradient descent 

    # If you want you could try Adam
#    optimizer = optim.Adam(model.parameters(), lr)

    # The loss function is the cross entropy.
    loss= torch.nn.CrossEntropyLoss(reduction= 'mean')     

    #Lists that will contain the information for each epoch
    loss_train= []
    acc_train=[]
    loss_test= []
    acc_test=[]
    num_epochs= int(epochs)

    # Training loop
    for epoch in range(1, num_epochs + 1):
  
        train_val, accuracy= model.backprop(inputs, targets , loss, epoch, optimizer)
        loss_train.append(train_val)
        acc_train.append(accuracy)        
            
        test_val, test_acc= model.test(test_inputs, test_targets, loss, epoch)       
        loss_test.append(test_val)
        acc_test.append(test_acc)

        if (epoch+1) % 2 == 0:
            print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                      '\tTraining Loss: {:.4f}'.format(train_val)+\
                      '\tTraining Accuracy: {:.4f}'.format(accuracy)+\
                      '\tTest Loss: {:.4f}'.format(test_val)+\
                      '\tTest Accuracy: {:.4f}'.format(test_acc))

    print('Final training loss: {:.4f}'.format(loss_train[-1]))
    print('Final training loss: {:.4f}'.format(acc_train[-1]))
    print('Final test loss: {:.4f}'.format(loss_test[-1]))
    print('Final training loss: {:.4f}'.format(acc_train[-1]))

    return  loss_train, acc_train, loss_test, acc_test

def plot_results(training, test, type_of_graph):
    assert len(training)==len(test), 'Length mismatch between the curves'
    num_epochs= len(training)

    # Plot saved in results folder
    plt.plot(range(num_epochs), training, label= "Training "+type_of_graph, color="blue")
    plt.plot(range(num_epochs), test, label= "Test "+type_of_graph, color= "green")
    plt.legend()
    plt.savefig(args.output + '/fig'+type_of_graph+'.pdf')
    plt.close()


if __name__ == '__main__':
    #Command line arguments
    parser= argparse.ArgumentParser(description='Inputs')
    
    parser.add_argument("-json", help="input path to json file")
    parser.add_argument("-input", default="data/even_mnist.csv" ,help="file path to datafile if not given then  data/even_mnist.csv is used")
    parser.add_argument("-output", default="results" ,help="file path to results")

    args = parser.parse_args()

    #Load data
    data=np.loadtxt(args.input)
    f=open(args.json)
    json_file=json.load(f)
    
    lr=json_file['learning rate']   # input 2a
    epochs=json_file['Epochs']    # input 2b
    n_hidden=json_file['n_hidden']  #number of hidden

    #Dimention 
    dim=data.shape
    n_s=dim[0]    #Number of samples
    i_d=dim[1]-1  #Dimensions of input
    n_bits=i_d    #number of bits

    #Creat an instance of the model
    model=Net(n_bits, n_hidden)
    
    #  Run the model 
    loss_train, acc_train, loss_test, acc_test=run_demo(lr, epochs, model, data)
    model.reset()

    #Plot and save
    plot_results(loss_train, loss_test, "loss")
    plot_results(acc_train, acc_test, "accuracy")