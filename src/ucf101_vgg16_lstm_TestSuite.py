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


def vgg16_lstm_test(r, criterion = "NC"):

    uvlc = ucf101_vgg16_lstm_class()
    uvlc.model.summary()
    
    
    r.resetTime()
    epsilon = 0.0001 


    if criterion == "NC": 
    
        layer1 = 0
        layer2 = 1
        
        
        nctoe = NCTestObjectiveEvaluation(r)
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
            b1 = 0
            b2 = 0.01
            # problem specific part 
            (label1,conf1) = uvlc.displayInfo(uvlc.getFunctors(uvlc.model),test)
            test2 = getNextInputByGradient(uvlc,epsilon,layer1,layer2,l2_norm,b1,l2_norm,b2,test,test,last_activation,0)
            print("found a test cases ...")

            (label2,conf2) = uvlc.displayInfo(uvlc.getFunctors(uvlc.model),test2)
            nctoe.updateSample(label2,label1)
            nctoe.testCase = test2
            nctoe.update_features()
            nctoe.writeInfo()
        
            if nctoe.coverage == 1.0 :
                print("statistics: ")  
                print("reach 100% coverage")
                nctoe.displayTrainingSamples()
                nctoe.displaySuccessRate()
                return
            else: 
                nctoe.testObjective.displayRemainingFeatures()

        print("statistics: ")
        nctoe.displayCoverage()
        nctoe.displayTrainingSamples()
        nctoe.displaySuccessRate()
        
    elif criterion == "MCDC": 

        layer1 = 0
        layer2 = 1
        

        mcdctoe = MCDCTestObjectiveEvaluation(r)
        mcdctoe.model = uvlc.model
        mcdctoe.testObjective.layer1 = layer1
        mcdctoe.testObjective.layer2 = layer2
        
        # problem specific part 
        images, preprocessed_images, test, last_activation = uvlc.predict(0)

        mcdctoe.testCase = test
        # set features for layer1
        activations1 = mcdctoe.get_activations1()
        mcdctoe.testObjective.feature1 = (np.argwhere(activations1 >= np.min(activations1))).tolist()
        mcdctoe.testObjective.setFeature1(10)
        mcdctoe.testObjective.setOriginalNumOfFeature1()
        #print("layer 1: %s"%(mcdctoe.testObjective.feature1))
        # set features for layer2
        activations2 = mcdctoe.get_activations2()
        mcdctoe.testObjective.feature2 = (np.argwhere(activations2 >= np.min(activations2))).tolist()
        mcdctoe.testObjective.setFeature2(10)
        mcdctoe.testObjective.setOriginalNumOfFeature2()
        #print("layer 2: %s"%(mcdctoe.testObjective.feature2))
        # set feature pairs
        mcdctoe.testObjective.initialisePairOfFeatures()
        #print("pairs: %s"%(mcdctoe.testObjective.pairOfFeatures))

        while len(mcdctoe.testObjective.pairOfFeatures) > 0: 
        
            (f1,f2) = mcdctoe.testObjective.pairOfFeatures[0] 
                    
            feature1 = mcdctoe.testObjective.feature1[f1]
            feature2 = mcdctoe.testObjective.feature2[f2]
            #print("feature 1: %s"%(feature1))
            #print("feature 2: %s"%(feature2))
        
            X = K.placeholder(ndim=2) #specify the right placeholder
            Y = K.sum(K.square(X)) # loss function
            fn = K.function([X], K.gradients(Y, [X])) #function to call the gradient    
            #get next input test2 from the current input test 
            b1 = 0.003
            b2 = 0.01
                        
            for index in range(1000): 
            
                # problem specific part 
                images, preprocessed_images, test, last_activation = uvlc.predict(index)
                mcdctoe.updateTrainingSample()

                (label1,conf1) = uvlc.displayInfo(uvlc.getFunctors(uvlc.model),test)
                test2 = getNextInputByGradientAndFeatures(uvlc,epsilon,layer1,layer2,l2_norm,b1,l2_norm,b2,feature1,feature2,test,test,last_activation,0)
                
                if not (test2 is None): 

                    (label2,conf2) = uvlc.displayInfo(uvlc.getFunctors(uvlc.model),test2)
                    mcdctoe.updateSample(label2,label1)
                    mcdctoe.testCase = test2
                    mcdctoe.update_features(f1,f2)
                    mcdctoe.writeInfo()
        
                    if mcdctoe.coverage == 1.0 :  
                        print("statistics: ")
                        print("reach 100% coverage")
                        mcdctoe.displayTrainingSamples()
                        mcdctoe.displaySuccessRate()
                        return
                    else: 
                        mcdctoe.testObjective.displayRemainingFeaturePairs()
                        
                    break
                    
        print("statistics: ")
        mcdctoe.displayCoverage()
        mcdctoe.displayTrainingSamples()
        mcdctoe.displaySuccessRate()
        
    else: print("coverage metric hasn't been developed.")
