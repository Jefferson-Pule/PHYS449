from matrix import Matrix
import numpy as np
class Graddescent_linear_regression(Matrix):
	#Initializa Class
	def __init__(self,x,y,dimension_of_each_input, learning_rate, repetitions):

		# Save our data
		self.x	 =	x						# input from experiment			
		self.y	 =	y						# output from experiment
		self.x_d =  dimension_of_each_input # dimensions of an input vector x
		self.lr  =  learning_rate			# Lerning rate
		self.epochs= repetitions			# Number of repetitions for the Grad descent

	def graddescent_linear_regression(self):
		
		x=self.x
		y=self.y
		x_d=self.x_d
		lr=self.lr
		epochs=self.epochs

		# We add a constant value x_0 to all our samples. This value can be chosen to be 1. 
		x=np.insert(x,0,1, axis=1)

		# Initial Guess for w:
		w_0=np.ones((x_d+1))

		#Start w
		w=w_0

		#Gradient descent loop

		for epoch in range(epochs):
			# Error y-approx(y)
			err=(y-np.sum(w*x,axis=1, keepdims=True))
			err_T=self.Tr(err)
			
			# Derivative
			dL=-np.reshape(self.Mult(err_T,x),w.shape)

			#Update w 
			w=w-lr*dL
		print("w_gs",w)

		return w


