import numpy as np
from keras import backend as K
import cv2
import os
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD

def lp_norm(p,n1,n2):
    return np.linalg.norm(np.array([n1]).ravel()-np.array([n2]).ravel(),ord=p) 
    
def l2_norm(n1,n2):
    return lp_norm(2,n1,n2)
    
def getActivationValue(model,layer,test):
    #print("xxxx %s"%(str(self.model.layers[1].input.shape)))
    OutFunc = K.function([model.input], [model.layers[layer].output])
    out_val = OutFunc([test, 1.])[0]
    return np.squeeze(out_val)
    
def layerName(model,layer):
    layerNames = [layer.name for layer in model.layers]
    return layerNames[layer]
    
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
def extract_vgg16_features(model, video_input_file_path, feature_output_file_path):
    if os.path.exists(feature_output_file_path):
        return np.load(feature_output_file_path)
    count = 0
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            input = img_to_array(img)
            input = np.expand_dims(input, axis=0)
            input = preprocess_input(input)
            feature = model.predict(input).ravel()
            features.append(feature)
            count = count + 1
    unscaled_features = np.array(features)
    np.save(feature_output_file_path, unscaled_features)
    return unscaled_features
    
def extract_vgg16_features_live(model, video_input_file_path):
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    images = []
    preprocessed_images = []
    success = True
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            images.append(image)
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            input = img_to_array(img)
            input = np.expand_dims(input, axis=0)
            input = preprocess_input(input)
            preprocessed_images.append(input[0])
            feature = model.predict(input).ravel()
            features.append(feature)
            count = count + 1
    unscaled_features = np.array(features)
    images = np.array(images)
    preprocessed_images = np.array(preprocessed_images)
    return images, preprocessed_images, unscaled_features
    
def scan_and_extract_vgg16_features(data_dir_path, output_dir_path, model=None, data_set_name=None):
    if data_set_name is None:
        data_set_name = 'UCF-101'

    input_data_dir_path = data_dir_path + '/' + data_set_name
    output_feature_data_dir_path = data_dir_path + '/' + output_dir_path

    if model is None:
        model = VGG16(include_top=True, weights='imagenet')
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    if not os.path.exists(output_feature_data_dir_path):
        os.makedirs(output_feature_data_dir_path)

    y_samples = []
    x_samples = []

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + os.path.sep + f
        if not os.path.isfile(file_path):
            output_dir_name = f
            output_dir_path = output_feature_data_dir_path + os.path.sep + output_dir_name
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                output_feature_file_path = output_dir_path + os.path.sep + ff.split('.')[0] + '.npy'
                x = extract_vgg16_features(model, video_file_path, output_feature_file_path)
                y = f
                y_samples.append(y)
                x_samples.append(x)

        if dir_count == MAX_NB_CLASSES:
            break

    return x_samples, y_samples
