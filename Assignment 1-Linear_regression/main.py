'''
PHYS 449 -- Fall 2021 
Assignment 1:
    -Linear regression by the exact method
    -Linear regression by gradient descent
'''
#Beginning of assignment

import numpy as np 
import json, argparse, sys, os

#Point to folders
sys.path.append("src")
sys.path.append("data")
my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
print(my_absolute_dirpath)
from linregexact import Exact_linear_regression
from linreggradds import Graddescent_linear_regression

if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser(description='Inputs')

    parser.add_argument("-input1", help="file path to datafile")
    parser.add_argument("-json", help="input path to json file")

    args = parser.parse_args()

    #name of data file
    name=os.path.splitext(args.input1)[0]
    
    # Load data
    data=np.loadtxt(args.input1) #input1
    # Read json file and extract 2 value
    f = open(args.json,)
    json_file = json.load(f)

    lr=json_file['learning rate']   # input 2a
    epochs=json_file['num iter']    # input 2b

    
    #Dimensions
    dim=data.shape
    n_d=dim[0]      #number of data points 
    x_d=dim[1]-1    # dimention of x
    
    # Data separation into inputs and outputs
    x=data[:,0:x_d]                 # x_values in a matrix
    y=data[:,[-1]]                  # y_values in a vector

    # Create instances of classes
    ex=Exact_linear_regression(x,y)
    gd=Graddescent_linear_regression(x,y,x_d, lr, epochs)
    
    # Execute exact linear regression and gradient descent linear regression to find w*.
    w_ex = ex.exact_linear_regression_function()
    w_gd = gd.graddescent_linear_regression()

    #Save the values of w_ex and w_gd on an .output file
    with open(name+".out","w") as out:
        np.savetxt(out, w_ex, delimiter='\n', fmt="%.4f")
        out.write("\n")
        np.savetxt(out, w_gd, delimiter='\n', fmt="%.4f")


    


