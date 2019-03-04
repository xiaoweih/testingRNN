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

    
def getNextInputByGradient(sm,epsilon,layer1,layer2,g1,b1,g2,b2,test1,test2,lastactivation2): 

    model = sm.model
    
    print("test shape: %s, lastactivation shape %s"%(str(test1.shape),str(lastactivation2.shape)))
    act1x = get_activations_single_layer(model,np.array([test1]),sm.layerName(layer1))
    act1y = get_activations_single_layer(model,np.array([test1]),sm.layerName(layer2))
    gd = get_gradients_of_activations_input(model,np.array([test2]),np.array([lastactivation2]),sm.layerName(0))
    gd = np.squeeze(list(gd.values())[0])
    newtest = np.round(test2[0] - epsilon * gd)
    act2x = get_activations_single_layer(model,np.array([newtest]),sm.layerName(layer1))
    act2y = get_activations_single_layer(model,np.array([newtest]),sm.layerName(layer2))
    lastactivation2 = get_activations_single_layer(model,np.array([newtest]),sm.layerName(-1))

    if g1(act1x,act2x) > b1 and g2(act1y,act2y) > b2: 
        print("found a test case of shape %s!"%(str(newtest.shape)))
        return newtest
    elif np.array_equal(test2,newtest):
        print("enlarge the epsilon by 10 times, and continue ... ")
        return getNextInput(sm,epsilon*10,layer1,layer2,g1,b1,g2,b2,test1,newtest,lastactivation2)
    else: 
        print("move a step away, and continue ... ")
        return getNextInput(sm,epsilon,layer1,layer2,g1,b1,g2,b2,test1,newtest,lastactivation2)
        
def getNextInputByGradientAndFeatures(sm,epsilon,layer1,layer2,g1,b1,g2,b2,feature1,feature2,test1,test2,lastactivation2): 

    model = sm.model
    
    print("test shape: %s, lastactivation shape %s"%(str(test1.shape),str(lastactivation2.shape)))
    act1x = get_activations_single_layer(model,np.array([test1]),sm.layerName(layer1))
    act1y = get_activations_single_layer(model,np.array([test1]),sm.layerName(layer2))
    act1x = getFeatureFromLayer(act1x,feature1)
    act1y = getFeatureFromLayer(act1y,feature2)
    
    gd = get_gradients_of_activations_input(model,np.array([test2]),np.array([lastactivation2]),sm.layerName(0))
    gd = np.squeeze(list(gd.values())[0])
    newtest = np.round(test2[0] - epsilon * gd)
    act2x = get_activations_single_layer(model,np.array([newtest]),sm.layerName(layer1))
    act2y = get_activations_single_layer(model,np.array([newtest]),sm.layerName(layer2))
    act2x = getFeatureFromLayer(act2x,feature1)
    act2y = getFeatureFromLayer(act2y,feature2)    

    lastactivation2 = get_activations_single_layer(model,np.array([newtest]),sm.layerName(-1))

    if g1(act1x,act2x) > b1 and g2(act1y,act2y) > b2: 
        print("found a test case of shape %s!"%(str(newtest.shape)))
        return newtest
    elif g1(act1x,act2x) > 10* b1 or g2(act1y,act2y) > 10 * b2: 
        print("failed to find a test case in this round! ")
        return None
    elif np.array_equal(test2,newtest):
        print("enlarge the epsilon by 10 times, and continue ... ")
        return getNextInput(sm,epsilon*10,layer1,layer2,g1,b1,g2,b2,test1,newtest,lastactivation2)
    else: 
        print("move a step away, and continue ... ")
        return getNextInput(sm,epsilon,layer1,layer2,g1,b1,g2,b2,test1,newtest,lastactivation2)



def getNextInputByCustomisedFunction(sm,fn,epsilon,layer1,layer2,g1,b1,g2,b2,test1,test2):
    model = sm.model 
    act1x = get_activations_single_layer(model,np.array([test1]),sm.layerName(layer1))
    act1y = get_activations_single_layer(model,np.array([test1]),sm.layerName(layer2))

    gd = fn([np.array([test2])])
    newtest = np.round(test2 - epsilon * gd[0][0])
    #newtestReview = sm.displayInfo(newtest)
    
    act2x = get_activations_single_layer(model,np.array([newtest]),sm.layerName(layer1))
    act2y = get_activations_single_layer(model,np.array([newtest]),sm.layerName(layer2))
        
    if g1(act1x,act2x) > b1 and g2(act1y,act2y) > b2: 
        print("found a test case of shape %s!"%(str(newtest.shape)))
        return newtest
    elif np.array_equal(test2,newtest):
        print("enlarge the epsilon by 10 times, and continue ... ")
        return getNextInputByCustomisedFunction(sm,fn,epsilon*10,layer1,layer2,g1,b1,g2,b2,test1,newtest)
    else: 
        print("move a step away, and continue ... ")
        return getNextInputByCustomisedFunction(sm,fn,epsilon,layer1,layer2,g1,b1,g2,b2,test1,newtest)
        
        
def getNextInputByCustomisedFunctionAndFeatures(sm,fn,epsilon,layer1,layer2,g1,b1,g2,b2,feature1,feature2,test1,test2):
    model = sm.model 
    act1x = get_activations_single_layer(model,np.array([test1]),sm.layerName(layer1))
    act1y = get_activations_single_layer(model,np.array([test1]),sm.layerName(layer2))
    #print("shape1: %s"%(str(act1x.shape)))
    #print("shape2: %s"%(str(act1y.shape)))
    act1x = getFeatureFromLayer(act1x,feature1)
    act1y = getFeatureFromLayer(act1y,feature2)

    gd = fn([np.array([test2])])
    newtest = np.round(test2 - epsilon * gd[0][0])
    #newtestReview = sm.displayInfo(newtest)
    
    act2x = get_activations_single_layer(model,np.array([newtest]),sm.layerName(layer1))
    act2y = get_activations_single_layer(model,np.array([newtest]),sm.layerName(layer2))
    act2x = getFeatureFromLayer(act2x,feature1)
    act2y = getFeatureFromLayer(act2y,feature2)
        
    if g1(act1x,act2x) > b1 and g2(act1y,act2y) > b2: 
        print("found a test case of shape %s!"%(str(newtest.shape)))
        return newtest
    elif g1(act1x,act2x) > 10* b1 or g2(act1y,act2y) > 10 * b2: 
        print("failed to find a test case in this round! ")
        return None
    elif np.array_equal(test2,newtest):
        print("enlarge the epsilon by 10 times, and continue ... ")
        return getNextInputByCustomisedFunction(sm,fn,epsilon*10,layer1,layer2,g1,b1,g2,b2,test1,newtest)
    else: 
        print("move a step away, and continue ... ")
        return getNextInputByCustomisedFunction(sm,fn,epsilon,layer1,layer2,g1,b1,g2,b2,test1,newtest)
        
        
def getFeatureFromLayer(activations,feature): 
    featureActivation = []
    for i in feature: 
        if len(activations.shape) == 1:
            l = i[0]
        elif len(activations.shape) == 2:
            l = (i[0],i[1])
        else:
            print("getFeatureFromLayer: more than can be handled ")
            exit()
        featureActivation.append(activations.item(l))
    return featureActivation
    

