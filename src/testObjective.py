import keras
from keras.datasets import imdb 
from keras.layers import *
from keras import *
from keras.models import *
import copy
import keras.backend as K
from keras.preprocessing import sequence 
from keract import * 
from utils import *


class NCTestObjectiveEvaluation: 

    def __init__(self): 
        self.testObjective = NCTestObjective()
        self.testCase = None
        self.model = None
        self.coverage = 1.0
        self.numAdv = 0
        self.numSamples = 0 
        self.numTrainingSample = 0
        
    def updateTrainingSample(self):
        self.numTrainingSample += 1
        
    def displayTrainingSamples(self):
        print("%s training samples are considered"%(self.numTrainingSample)) 
        
    def setTestCase(self,testCase):
        self.testCase = testCase 
            
    def setModel(self,model):
        self.model = model
        
    def get_activations(self): 
        return get_activations_single_layer(self.model,np.array([self.testCase]),layerName(self.model,self.testObjective.layer))
        
    def update_features(self): 
        activation = self.get_activations()
        features = (np.argwhere(activation > 0)).tolist()
        print("found %s features."%(len(features)))
        #print("including %s."%(str(features)))
        for feature in features: 
            if feature in self.testObjective.feature: 
                self.testObjective.feature.remove(feature) #np.delete(self.testObjective.feature, np.where(self.testObjective.feature == feature), axis=0)

        self.coverage = 1 - len(self.testObjective.feature)/self.testObjective.originalNumOfFeature
        self.displayCoverage()
        
    def displayCoverage(self):
        print("coverage up to now: %s\n"%(self.coverage))
                  
    def evaluate(self): 
        if self.testObjective.feature == []: 
            return True
        else: return False
                
    def updateSample(self,conf): 
        if conf > 0.5: self.numAdv += 1
        self.numSamples += 1
        
    def displaySuccessRate(self): 
        print("%s samples, within which there are %s adversarial examples"%(self.numSamples,self.numAdv))
        print("the rate of adversarial examples is %s"%(self.numAdv/self.numSamples))

class NCTestObjective:
    def __init__(self):
        self.layer = None
        self.feature = None
        self.originalNumOfFeature = None

    def setOriginalNumOfFeature(self): 
        self.originalNumOfFeature = len(self.feature)
        self.displayRemainingFeatures()
        
    def displayRemainingFeatures(self):
        print("%s features to be covered."%(self.originalNumOfFeature))
        #print("including %s."%(str(self.feature)))


    def setLayer(self,layer):
        self.layer = layer 
            
    def setFeature(self,feature):
        self.feature = feature
        
