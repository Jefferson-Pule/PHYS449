from matrix import Matrix
import numpy as np
#Inhehrite Matrix Operations

class Exact_linear_regression(Matrix):
	#Initializa Class
	def __init__(self,x,y):
		#Save our data
		self.x	=	x		# input from experiment			
		self.y	=	y		# output from experiment
		
	def exact_linear_regression_function(self):
		
		x=self.x
		y=self.y

		# We add a constant value x_0 to all our samples. This value can be chosen to be 1. 
		x=np.insert(x,0,1, axis=1)
		print("x plus the neutral value",x)
		
		x_T=self.Tr(x)
		w=self.Mult(self.Mult(self.Inv(self.Mult(x_T,x)),x_T),y)
		w=np.reshape(w, w.shape[0])
		print("w_exact",w)

		return w





