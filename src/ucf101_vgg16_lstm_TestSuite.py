import numpy as np
from keras import backend as K
import sys
import os
import operator
import cv2
from keras.preprocessing.image import img_to_array
from testCaseGeneration import *
from ucf101_vgg16_lstm_class import *
from utils import lp_norm
from testObjective import *

K.set_learning_phase(1)
K.set_image_dim_ordering('tf')


def vgg16_lstm_test(criterion = "NC"):

    uvlc = ucf101_vgg16_lstm_class()
    uvlc.model.summary()

    if criterion == "NC": 
    
        layer1 = 0
        layer2 = 1
        
        epsilon = 0.0001 
        
        nctoe = NCTestObjectiveEvaluation()
        nctoe.model = uvlc.model
        nctoe.testObjective.layer = layer2
        
        # problem specific part 
        images, preprocessed_images, test, last_activation = uvlc.predict(0)
        
        nctoe.testCase = test
        activations = nctoe.get_activations()
        nctoe.testObjective.feature = (np.argwhere(activations >= np.min(activations))).tolist()
        nctoe.testObjective.setOriginalNumOfFeature()
    
        for index in range(1,100): 
        
            nctoe.updateTrainingSample()
            
            # problem specific part 
            images, preprocessed_images, test, last_activation = uvlc.predict(index)
    
            # debugging purpose
            uvlc.displayInfo(uvlc.getFunctors(uvlc.model),test)
            features = uvlc.model.predict(np.array([uvlc.predictor.vgg16_model.predict(preprocessed_images)])).ravel()
            uvlc.displayInfo(uvlc.getFunctors(uvlc.model),uvlc.predictor.vgg16_model.predict(preprocessed_images))  
    
            print("start finding test cases ...")
            b1 = 0.01
            b2 = 0.01
            # problem specific part 
            test2 = getNextInputByGradient(uvlc,epsilon,layer1,layer2,l2_norm,b1,l2_norm,b2,test,test,last_activation)
            print("found a test cases ...")

            conf = uvlc.displayInfo(uvlc.getFunctors(uvlc.model),test2)
            nctoe.updateSample(conf)
            nctoe.testCase = test2
            nctoe.update_features()
        
            if nctoe.coverage == 1.0 :  
                print("reach 100% coverage")
                nctoe.displayTrainingSamples()
                nctoe.displaySuccessRate()
                exit()
            else: 
                nctoe.testObjective.displayRemainingFeatures()

        nctoe.displayCoverage()
        nctoe.displayTrainingSamples()
        nctoe.displaySuccessRate()
        
    else: print("coverage metric hasn't been developed.")
