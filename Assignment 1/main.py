'''
PHYS 449 -- Fall 2021 
Assignment 1:
    -Linear regression by the exact method
    -Linear regression by gradient descent
'''
#Beginning of assignment

import numpy as np 
import json, argparse, sys

#Point to folders
sys.path.append("src")
sys.path.append("data")

from linregexact import Exact_linear_regression
from linreggradds import Graddescent_linear_regression

if __name__ == '__main__':


    # Command line arguments
    parser = argparse.ArgumentParser(description='Inputs')

    parser.add_argument('-input1', help='file path to datafile')
    parser.add_argument('-json', help='input path to json file')

    args = parser.parse_args()
    print(args.input1, args.json) #string input

    #read json file and extract 2 value
    f = open(args.json,)
 
    json_file = json.load(f)
    print(json_file)

    data=np.loadtxt(args.input1) #input1

    dim=data.shape
    #number of data points
    n_d=dim[0]

    #dimention of x
    x_d=dim[1]-1
    x=data[:,0:x_d]
    y=data[:,[-1]]
    lr=json_file['learning rate']#0.0001 #input 2a
    epochs=json_file['num iter']#200000 #input 2b

    ex=Exact_linear_regression(x,y)
    gd=Graddescent_linear_regression(x,y,x_d, lr, epochs)

    m=ex.Tr(x)
    print(m)
    w_ex=ex.exact_linear_regression_function()
    w_gd=gd.graddescent_linear_regression()


