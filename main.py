import argparse
import keras
from keras.datasets import imdb 
from keras.layers import *
from keras import *
from keras.models import *
import copy
import keras.backend as K

import sys
sys.path.append('example')
sys.path.append('src')

from sentimentTestSuite import sentimentTrainModel, sentimentGenerateTestSuite
from ucf101_vgg16_lstm_TestSuite import vgg16_lstm_test


def main(train=False):

    #modelName = "sentiment"
    
    parser=argparse.ArgumentParser(description='testing for recurrent neural networks' )
    parser.add_argument('--model', dest='modelName', default='sentiment', help='')
    parser.add_argument('--criterion', dest='criterion', default='NC', help='')

    args=parser.parse_args()
    
    modelName = args.modelName
    criterion = args.criterion

    if modelName == 'sentiment': 
        if train: 
            sentimentTrainModel()
        else: 
            sentimentGenerateTestSuite(criterion)

    elif modelName == 'ucf101': 
        vgg16_lstm_test(criterion)
        
    else: 
        print("Please specify a model from {sentiment, UCF101}")
    
if __name__=="__main__":
  main()