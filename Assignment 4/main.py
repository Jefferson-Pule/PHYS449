'''
PHYS 449 -- Fall 2021 
Assignment 4:
    -Fully visible RBM
'''
#Beginning of assignment

import numpy as np 
import json, argparse, sys, os

#Point to folders
sys.path.append("src")
sys.path.append("data")
my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
print(my_absolute_dirpath)

def get_data(file_path):
    
    raw_data=np.loadtxt(file_path,dtype=np.str)
    data=[]
    for i in range(raw_data.shape[0]):
        raw_sample=raw_data[i]
        spins=[]
        for spin in raw_sample:
            if spin=="+":
                spins.append(1)
            else:
                spins.append(0)
        data.append(spins)
    return np.array(data)


from RBM import RBM

if __name__ == '__main__':

    # Command line arguments
    #Command line arguments
    parser= argparse.ArgumentParser(description='Inputs')
    
    parser.add_argument("-json", default="data/arguments.json" ,help="input path to json file. Default data/arguments.json")
    parser.add_argument("-input", default="data/in.txt" ,help="file path to datafile. Default  data/in.txt ")
    parser.add_argument("-output", default="results" ,help="file path to results. Default results")
    parser.add_argument("-verbosity", default="4" ,help="After how many epochs to show the loss function")

    args = parser.parse_args()

    #Input data
    data_path=args.input
    ver=int(args.verbosity)
    f=open(args.json)
    json_file=json.load(f)
    
    lr=json_file['learning rate']       
    epochs=json_file['Epochs']    
    k=json_file['iterations_for_MH']    # Number of Iterations for Metropolis-Hastings

    
    # Transform data to numbers

    data=get_data(data_path)
    N=data.shape[1]                     # Size of our system
    
    # Create an instance of the model
    rbm=RBM(N,k)

    #Training the RBM 
    J,loss=rbm.train(epochs,data, lr, ver)
    J=J.numpy()
    print("J",J)
    results={}
    for n in range(len(rbm.nearest_neighbor)):
        results["("+str(rbm.nearest_neighbor[n][0])+", "+str(rbm.nearest_neighbor[n][1])+")"]=J[n]
    print("dictionary",results)

    with open(args.output+"/results.txt", "w") as file:
        print(results, file=file)




    


