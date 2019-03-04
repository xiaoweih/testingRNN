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
from record import record


def main(train=False):
    
    parser=argparse.ArgumentParser(description='testing for recurrent neural networks' )
    parser.add_argument('--model', dest='modelName', default='sentiment', help='')
    parser.add_argument('--criterion', dest='criterion', default='NC', help='')
    parser.add_argument('--mode', dest='mode', default='test', help='')
    parser.add_argument('--output', dest='filename', default='record.txt', help='')
    
    args=parser.parse_args()
    
    modelName = args.modelName
    criterion = args.criterion
    mode = args.mode
    filename = args.filename
    
    r = record(filename,time.time())

    if modelName == 'sentiment': 
        if mode == 'train': 
            sentimentTrainModel()
        else: 
            sentimentGenerateTestSuite(r,criterion)

    elif modelName == 'ucf101': 
        if mode == 'train': 
            vgg16_lstm_train()
        else: 
            vgg16_lstm_test(r,criterion)
        
    else: 
        print("Please specify a model from {sentiment, UCF101}")
    
    r.close()
    
if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))