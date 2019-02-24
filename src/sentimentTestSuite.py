import keras
from keras.datasets import imdb 
from keras.layers import *
from keras import *
from keras.models import *
import copy
import keras.backend as K
from keras.preprocessing import sequence 

from sentimentClass import Sentiment
from testCaseGeneration import *
from utils import lp_norm, powerset
from testObjective import *

def sentimentTrainModel(): 

    sm = Sentiment()
    sm.train_model()

def sentimentGenerateTestSuite(criterion = "NC"):

    sm = Sentiment()
    sm.load_model()
    
    if criterion == "NC": 

        layer1 = 0
        layer2 = 1

        nctoe = NCTestObjectiveEvaluation()
        nctoe.model = sm.model
        nctoe.testObjective.layer = layer2
        
        #predict sentiment from reviews
        review = "i really liked the movie and had fun"
        tmp = sm.fromTextToID(review)
        test = np.squeeze(sm.pre_processing_x(tmp))
        
        #lstm_states = sm.get_lstm_state(test)
        #print("%s"%(str(lstm_states.shape)))

        nctoe.testCase = test
        activations = nctoe.get_activations()
        nctoe.testObjective.feature = (np.argwhere(activations >= np.min(activations))).tolist()
        nctoe.testObjective.setOriginalNumOfFeature()

        epsilon = 0.0001 

        for test in sm.X_train :
        
            nctoe.updateTrainingSample()
            X = K.placeholder(ndim=2) #specify the right placeholder
            Y = K.sum(K.square(X)) # loss function
            fn = K.function([X], K.gradients(Y, [X])) #function to call the gradient    
            #get next input test2 from the current input test 
            b1 = 0.01
            b2 = 0.01
            test2 = getNextInputByCustomisedFunction(sm,fn,epsilon,layer1,layer2,l2_norm,b1,l2_norm,b2,test,test)
            conf = sm.displayInfo(test2)
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
                
            #test = np.squeeze(test)
            #last_activation = sm.model.predict(np.array([test]))
            #test2 = getNextInputByGradient(sm,epsilon,0,1,lp_norm,b1,lp_norm,b2,test,test,last_activation[0])

    
            #states = np.array([K.zeros((dim,32)) for dim in [units,units]])
            #print("state shape=%s    %s"%(str(states.shape), str(test.shape)))
            #h, [h, c] = step(test, states, W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, b_i, b_f, b_c, b_o)
            
    else: print("coverage metric hasn't been developed.")
    
    
