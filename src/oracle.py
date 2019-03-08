import numpy as np
import os
import time
from utils import lp_norm

class oracle: 

    def __init__(self,input,lp,radius): 

        self.input = input
        self.measurement = lp
        self.radius = radius

        
    def passOracle(self,test): 
       return np.linalg.norm(self.input - test,ord=self.measurement) <= radius 

    def measure(self,test): 
        return np.linalg.norm(self.input - test,ord=self.measurement)