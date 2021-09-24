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

    data=np.loadtxt("data/1.in")
    dim=data.shape
    #number of data points
    n_d=dim[0]

    #dimention of x
    x_d=dim[1]-1
    x=data[:,0:x_d]
    y=data[:,[-1]]
    lr=0.0001
    epochs=200000

    ex=Exact_linear_regression(x,y)
    gd=Graddescent_linear_regression(x,y,x_d, lr, epochs)

    m=ex.Tr(x)
    print(m)
    w_ex=ex.exact_linear_regression_function()
    w_gd=gd.graddescent_linear_regression()


