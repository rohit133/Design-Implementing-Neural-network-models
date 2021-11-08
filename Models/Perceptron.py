# Importing the Dependencies 
from numpy.core.numerictypes import ScalarType
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()

class Perceptron(object):
    def __init__(self, input_dimension):
        self.weight = np.random.normal(loc = 0.0, scale= 1.0, size = input_dimension)
        self.bias = np.random.normal(loc = 0.0, scale = 1.0, size = 1)
    
    def activation_relu(self,x):
        if x>0:
            return x
        else:
            return 0
        
    def predict(self, data):
        return self.activation_relu(np.dot(data, self.weight))
    
    def training(self, input_data, targets, epochs, lr = 0.01):
        loss = []
        for e in range(1, epochs + 1):
            epoch_loss = 0
            for data,target in zip(input_data,targets):
                pred = self.predict(data)
                error = target - pred
                update = lr * error
                epoch_loss += error ** 2
                self.weight += update * data
                self.bias += update
            loss.append(epoch_loss)

            print(f"\rEpoch : {e}/{epochs}, loss : {epoch_loss}")   
        return loss

x = data.data
y = data.target
obj = Perceptron(input_dimension = 4)
losses = obj.training(x,y,epochs= 100)
