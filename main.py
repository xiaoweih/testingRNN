import argparse
import keras
from keras.datasets import imdb 
from keras.layers import *
from keras import *
from keras.models import *
import copy
import keras.backend as K
import time

import sys
sys.path.append('example')
sys.path.append('src')

from sentimentTestSuite import sentimentTrainModel, sentimentGenerateTestSuite
from ucf101_vgg16_lstm_TestSuite import vgg16_lstm_test
from vgg16_lstm_train import vgg16_lstm_train


def main(train=False):
    
    parser=argparse.ArgumentParser(description='testing for recurrent neural networks' )
    parser.add_argument('--model', dest='modelName', default='sentiment', help='')
    parser.add_argument('--criterion', dest='criterion', default='NC', help='')
    parser.add_argument('--mode', dest='mode', default='test', help='')

    args=parser.parse_args()
    
    modelName = args.modelName
    criterion = args.criterion
    mode = args.mode

    if modelName == 'sentiment': 
        if mode == 'train': 
            sentimentTrainModel()
        else: 
            sentimentGenerateTestSuite(criterion)

    elif modelName == 'ucf101': 
        if mode == 'train': 
            vgg16_lstm_train()
        else: 
            vgg16_lstm_test(criterion)
        
    else: 
        print("Please specify a model from {sentiment, UCF101}")
    
if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))