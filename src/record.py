import numpy as np
import os
import time

class record: 

    def __init__(self,filename,startTime): 

        self.startTime = startTime

        directory = os.path.dirname(filename)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory) 
        self.file = open(filename,"w+") 
        
    def write(self,text): 
        self.file.write(text) 
        
    def close(self): 
        self.file.close()

    def resetTime(self): 
        self.write("reset time at %s\n\n"%(time.time() - self.startTime))
        self.startTime = time.time()
