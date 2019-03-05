import numpy as np
from keras import backend as K
import sys
import os
import operator
import cv2
from keras.preprocessing.image import img_to_array
from recurrent_networks import VGG16LSTMVideoClassifier
from UCF101_loader import load_ucf, scan_ucf_with_labels
    
K.set_learning_phase(1)


epsilon = 0.0001 


class ucf101_vgg16_lstm_class:

    def __init__(self):
        self.vgg16_include_top = True
        self.data_dir_path = os.path.join('dataset', 'very_large_data')
        self.model_dir_path = os.path.join('models', 'UCF-101')
        self.config_file_path = VGG16LSTMVideoClassifier.get_config_file_path(self.model_dir_path, vgg16_include_top=self.vgg16_include_top)
        self.weight_file_path = VGG16LSTMVideoClassifier.get_weight_file_path(self.model_dir_path, vgg16_include_top=self.vgg16_include_top)
    
        self.predictor = VGG16LSTMVideoClassifier()
        self.model = self.predictor.load_model(self.config_file_path, self.weight_file_path)
        
        self.videos = scan_ucf_with_labels(self.data_dir_path, [label for (label, label_index) in self.predictor.labels.items()])

        self.video_file_path_list = np.array([file_path for file_path in self.videos.keys()])
        np.random.shuffle(self.video_file_path_list)
    
    def predict(self,index=1): 
        video_file_path = self.video_file_path_list[index]
        label = self.videos[video_file_path]
        images, preprocessed_images, test, predicted_label, predicted_confidence, last_activation = self.predictor.predict(video_file_path)
        print("original image sequence shape: %s, preprocessed image sequence shape %s, feature sequence shape %s"%(str(images.shape),str(preprocessed_images.shape),str(test.shape)))
        print('predicted: ' + predicted_label + ' confidence:' + str(predicted_confidence) + ' actual: ' + label)
        return images, preprocessed_images, test, last_activation
        
    def getFunctors(self,model):
        outputs = [layer.output for layer in model.layers]          # all layer outputs
        functors = [K.function([model.input, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
        return functors

    def displayInfo(self,functors,test):
        test = np.array([test])
        conf = functors[-1]([test])[0][0]
        #print("current confidence: %s"%(conf))
        index, value = max(enumerate(conf), key=operator.itemgetter(1))
        print("the label is: %s, with confidence %s"%(self.predictor.labels_idx2word[index],value))
        return (index,value)

    def layerName(self,layer):
        layerNames = [layer.name for layer in self.model.layers]
        return layerNames[layer]



