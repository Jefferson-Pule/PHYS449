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

# Loss Function
def vae_loss(inputs ,mean, log_sigma, outputs, ns):

    inputs=inputs.reshape(ns,n_bits*n_bits)
    outputs=outputs.reshape(ns,n_bits*n_bits)
    
    # I used |x-x_lambda|**2 as my rec_likelihood
    rec_likehood=(torch.sum((inputs-outputs)**2, dim=1))

    # Calculate the KL divergence using properties of Normal distributions

    kl_div=-1-log_sigma+torch.square(mean)+torch.exp(log_sigma)
    kl_div=torch.sum(kl_div, dim=-1)
    kl_div*=0.5
    loss=torch.mean(rec_likehood+kl_div)

    return loss

def run_demo(lr,epochs, model, data, ns):

    # Divide data into inputs and lables, but also into a 3000 test 
    inputs= model.getdata(data)

    #Transform the numpy arrays into tensors
    inputs= torch.from_numpy(inputs)
    inputs=inputs.float()

    # Define an optimizer and the loss function
    
    optimizer = optim.Adam(model.parameters(), lr)  # We use gradient descent 

    #Lists that will contain the information for each epoch
    loss_train= []
    num_epochs= int(epochs)

    # Training loop
    for epoch in range(1, num_epochs + 1):
  
        train_val = model.backprop(inputs, vae_loss, epoch, optimizer, ns)
        loss_train.append(train_val)
        if verb:
            if (epoch+1) % verb_epoch == 0:
                print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                          '\tTraining Loss: {:.4f}'.format(train_val))

    print('Final training loss: {:.4f}'.format(loss_train[-1]))

    return  loss_train

def plot_mnist(inputs, outputs, n_bits):

    for n in range(len(inputs)):
        fig, axarr = plt.subplots(1,2)
        axarr[0].imshow(np.reshape(inputs[n].detach().numpy(), (n_bits,n_bits)))
        axarr[1].imshow(np.reshape(outputs[n].detach().numpy(), (n_bits,n_bits)))
        
        axarr[0].title.set_text('Data')
        axarr[1].title.set_text('Generated')

        plt.savefig(args.output + '/'+str(n)+'.pdf')
        plt.close()


def plot_results(training, type_of_graph):

    num_epochs= len(training)

    # Plot saved in results folder
    plt.plot(range(num_epochs), training, label= "Training "+type_of_graph, color="blue")
    plt.legend()
    plt.savefig(args.output + '/fig'+type_of_graph+'.pdf')
    plt.close()


if __name__ == '__main__':
    #Command line arguments
    parser= argparse.ArgumentParser(description='Inputs')

    parser.add_argument("-n", default="100" ,help="number of samples to be drawn at the end, default 100")
    parser.add_argument("-json", default="data/arguments.json" ,help="input path to json file with training param, default data/arguments.json")
    parser.add_argument("-input", default="data/even_mnist.csv" ,help="file path to datafile, default data/even_mnist.csv is used")
    parser.add_argument("-output", default="results" ,help="file path to results")
    parser.add_argument("-verbose", default="True" ,help="Verbose mode True or False. default True")
    parser.add_argument("-verbose_epoch", default="2" ,help="After how many epochs to give the report, default 2")

    args = parser.parse_args()
    # Create directory results if it does not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #Load data
    data=np.loadtxt(args.input)
    f=open(args.json)
    json_file=json.load(f)
    
    lr=json_file['learning rate']  
    epochs=json_file['Epochs']    
    lattent_dims=json_file['Lattent_dims']

    # Number of samples to draw at the end   
    n=int(args.n)

    # Verbose
    verb=bool(args.verbose)
    verb_epoch=int(args.verbose_epoch)

    #Dimention 
    dim=data.shape
    n_s=dim[0]    #Number of samples
    i_d=dim[1]-1  #Dimensions of input
    n_bits=int(np.sqrt(i_d))    #number of bits


    #Creat an instance of the model
    model=Net(n_bits, lattent_dims)
    
    #  Run the model 
    loss_train=run_demo(lr, epochs, model, data, n_s)

    #Plot and save
    plot_results(loss_train,  "loss")

    # Generate 100 images

    inputs=model.getdata(data)

    # Choose random index to sample from inputs
    
    index=np.arange(0,n_s,1)
    samples_index=np.random.choice(index, n, replace=False)
    test_samples=np.take(inputs, samples_index, axis=0)
    test_samples=torch.Tensor(test_samples)
    
    # Genearete an output 
    _,__,test_output=model.forward(test_samples,n)
    
    plot_mnist(test_samples, test_output, n_bits)

#    model.reset()