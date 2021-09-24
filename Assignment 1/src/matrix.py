import numpy as np

#Redefine Matrix Operations
class Matrix:

    #Transpose 
    def Tr(self,x):
        return np.matrix.transpose(x)

    # Inverse
    def Inv(self,x):
        return np.linalg.inv(x)    

    # Matrix product
    def Mult(self,x,y):
        return np.matmul(x,y)
    
    #dot product
    def Dot(self,x,y):
        return x*y


