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
from random import *
import numpy as np
import time

class MCDCTestObjectiveEvaluation: 

    def __init__(self,r): 
        self.testObjective = MCDCTestObjective()
        self.testCase = None
        self.model = None
        self.coverage = 1.0
        self.numAdv = 0
        self.numSamples = 0 
        self.numTrainingSample = 0
        self.record = r
        self.perturbations = []
        
    def updateTrainingSample(self):
        self.numTrainingSample += 1
        
    def displayTrainingSamples(self):
        print("%s training samples are considered"%(self.numTrainingSample)) 
        
    def setTestCase(self,testCase):
        self.testCase = testCase 
            
    def setModel(self,model):
        self.model = model
        
    def get_activations1(self): 
        return get_activations_single_layer(self.model,np.array([self.testCase]),layerName(self.model,self.testObjective.layer1))
        
    def get_activations2(self): 
        return get_activations_single_layer(self.model,np.array([self.testCase]),layerName(self.model,self.testObjective.layer2))
        
    def update_features(self,f1,f2): 
        #print("all: %s "%(self.testObjective.pairOfFeatures))
        #print("now %s"%(str((f1,f2))))
        self.testObjective.pairOfFeatures.remove((f1,f2))
        self.coverage = 1 - len(self.testObjective.pairOfFeatures)/self.testObjective.originalNumOfPairOfFeatures
        self.displayCoverage()
        
    def displayCoverage(self):
        print("coverage up to now: %.2f\n"%(self.coverage))
                  
    def evaluate(self): 
        if self.testObjective.feature == []: 
            return True
        else: return False
                
    def updateSample(self,label2,label1,m): 
        if label2 != label1: 
            self.numAdv += 1
            self.perturbations.append(m)
        self.numSamples += 1
        self.displaySuccessRate()
        
    def displaySamples(self):
        print("%s samples are considered"%(self.numSamples)) 
        
    def displaySuccessRate(self): 
        print("%s samples, within which there are %s adversarial examples"%(self.numSamples,self.numAdv))
        print("the rate of adversarial examples is %.2f"%(self.numAdv/self.numSamples))
        
    def displayPerturbations(self):
        print("the average perturbation of the adversarial examples is %s"%(sum(self.perturbations)/self.numAdv))
        print("the smallest perturbation of the adversarial examples is %s"%(min(self.perturbations)))

    def writeInfo(self): 
        self.record.write("time:%s\n"%(time.time() - self.record.startTime))
        self.record.write("training samples: %s\n"%(self.numTrainingSample))
        self.record.write("samples: %s\n"%(self.numSamples))
        self.record.write("coverage: %.2f\n"%(self.coverage))
        self.record.write("success rate: %.2f\n"%(self.numAdv/self.numSamples))
        self.record.write("average perturbation: %.2f\n"%(sum(self.perturbations)/self.numAdv))
        self.record.write("minimum perturbation: %.2f\n\n"%(min(self.perturbations)))

class MCDCTestObjective:
    def __init__(self):
        self.layer1 = None
        self.feature1 = None
        self.layer2 = None
        self.feature2 = None
        self.originalNumOfFeature1 = None
        self.originalNumOfFeature2 = None
        self.originalNumOfPairOfFeatures = None
        self.pairOfFeatures = None

    def setOriginalNumOfFeature1(self): 
        self.originalNumOfFeature1 = len(self.feature1)
        self.displayRemainingFeatures1()
        
    def displayRemainingFeatures1(self):
        print("%s features to be covered in layer %s."%(self.originalNumOfFeature1, self.layer1))

    def setLayer1(self,layer1):
        self.layer1 = layer1 
            
    def setFeature1(self,n):
        features = {}
        for i in range(n):
                features[i] = self.getRandomSubsetOfFeature(self.feature1)
        self.feature1 = features
        
    def setOriginalNumOfFeature2(self): 
        self.originalNumOfFeature2 = len(self.feature2)
        self.displayRemainingFeatures2()
        
    def displayRemainingFeatures2(self):
        print("%s features to be covered in layer %s."%(self.originalNumOfFeature2, self.layer2))
        
    def displayRemainingFeaturePairs(self):
        print("Totally there are %s feature pairs, within which %s feature pairs are to be covered."%(self.originalNumOfPairOfFeatures,len(self.pairOfFeatures)))

    def setLayer2(self,layer2):
        self.layer2 = layer2 
            
    def setFeature2(self,n):
        features = {}
        for i in range(n):
                features[i] = self.getRandomSubsetOfFeature(self.feature2)
        self.feature2 = features
        
    def initialisePairOfFeatures(self):
        import itertools
        if self.originalNumOfFeature1 > 0 and self.originalNumOfFeature2 > 0: 
            self.pairOfFeatures = list(itertools.product(self.feature1.keys(),self.feature2.keys()))
            self.originalNumOfPairOfFeatures = len(self.pairOfFeatures)
            print("%s pairs of features needed to be tested."%(self.originalNumOfPairOfFeatures))
        else: 
            print("cannot initialise feature pairs")
            exit()
            
    def getRandomSubsetOfFeature(self,features): 
        left = randint(0, len(features) - 1)
        right = randint(left + 1, len(features))
        return features[left:right]


class NCTestObjectiveEvaluation: 

    def __init__(self,r): 
        self.testObjective = NCTestObjective()
        self.testCase = None
        self.model = None
        self.coverage = 1.0
        self.numAdv = 0
        self.numSamples = 0 
        self.numTrainingSample = 0
        self.record = r
        self.perturbations = []

        
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
        print("coverage up to now: %.2f\n"%(self.coverage))
                  
    def evaluate(self): 
        if self.testObjective.feature == []: 
            return True
        else: return False
                
    def updateSample(self,label2,label1,m): 
        if label2 != label1: 
            self.numAdv += 1
            self.perturbations.append(m)
        self.numSamples += 1
        self.displaySuccessRate()
        
    def displaySamples(self):
        print("%s samples are considered"%(self.numSamples)) 
        
    def displaySuccessRate(self): 
        print("%s samples, within which there are %s adversarial examples"%(self.numSamples,self.numAdv))
        print("the rate of adversarial examples is %.2f"%(self.numAdv/self.numSamples))
        
    def displayPerturbations(self):
        print("the average perturbation of the adversarial examples is %s"%(sum(self.perturbations)/self.numAdv))
        print("the smallest perturbation of the adversarial examples is %s"%(min(self.perturbations)))
        
    def writeInfo(self): 
        self.record.write("time:%s\n"%(time.time() - self.record.startTime))
        self.record.write("training samples: %s\n"%(self.numTrainingSample))
        self.record.write("samples: %s\n"%(self.numSamples))
        self.record.write("coverage: %.2f\n"%(self.coverage))
        self.record.write("success rate: %.2f\n\n"%(self.numAdv/self.numSamples))
        if self.numAdv > 0 : self.record.write("average perturbation: %.2f\n"%(sum(self.perturbations)/self.numAdv))
        self.record.write("minimum perturbation: %.2f\n\n"%(min(self.perturbations)))

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
        
