import keras
from keras.datasets import imdb 
from keras.layers import *
from keras import *
from keras.models import *
import copy
import keras.backend as K
from keras.preprocessing import sequence 
from utils import getActivationValue,layerName
from keract import get_activations_single_layer

class Sentiment:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        
        self.top_words = 5000
        self.word_to_id = keras.datasets.imdb.get_word_index()
        self.INDEX_FROM=3
        self.max_review_length = 500
        self.embedding_vector_length = 32 
        
        self.word_to_id = {k:(v+self.INDEX_FROM) for k,v in self.word_to_id.items()}
        self.word_to_id["<PAD>"] = 0
        self.word_to_id["<START>"] = 1
        self.word_to_id["<UNK>"] = 2
        self.id_to_word = {value:key for key,value in self.word_to_id.items()}
        
        self.load_data()
        self.pre_processing_X()
        
        self.construct_lstm_state_model()
        
    
    def load_data(self): 
        (self.X_train, self.y_train), (self.X_test, self.y_test) = imdb.load_data(num_words=self.top_words)
        
    def load_model(self):
        self.model=load_model('models/sentiment-lstm2.h5')
        self.model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
        self.model.summary()
        
    def layerName(self,layer):
        layerNames = [layer.name for layer in self.model.layers]
        return layerNames[layer]
        
    def construct_lstm_state_model(self):
        inputs1 = Input(shape=(500,))
        embedding1 = Embedding(self.top_words, self.embedding_vector_length, input_length=self.max_review_length, input_shape=(self.top_words,))(inputs1)
        lstm1, state_h, state_c = LSTM(100, return_sequences=True, return_state=True)(embedding1)
        self.lstm_state_model = Model(inputs=inputs1, outputs = [lstm1,state_h,state_c])
        self.lstm_state_model.summary()
        
    def get_lstm_state(self,test):
        return get_activations_single_layer(self.lstm_state_model,np.array([test]),'lstm_1')
        #return self.lstm_state_model.predict(np.array([test]))[0]
        
    def train_model(self):
        self.load_data()
        self.pre_processing_X()
        self.model = Sequential() 
        self.model.add(Embedding(self.top_words, self.embedding_vector_length, input_length=self.max_review_length, input_shape=(self.top_words,))) 
        self.model.add(LSTM(100)) 
        self.model.add(Dense(1, activation='sigmoid')) 
        self.model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
        print(self.model.summary()) 
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), nb_epoch=10, batch_size=64) 
        scores = model.evaluate(self.X_test, self.y_test, verbose=0) 
        print("Accuracy: %.2f%%" % (scores[1]*100))
        model.save('models/sentiment-lstm2.h5')
                
    def getFunctors(self,model):
        outputs = [layer.output for layer in model.layers]          # all layer outputs
        functors = [K.function([model.input, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
        return functors

    def getOutputResult(self,model,test):
        functors = self.getFunctors(model)
        return functors[-1]([test, 1.])[0][0]

    def displayInfo(self,test):
        model = self.model
        text = self.fromIDToText(test)
        print("review content: %s"%(self.fromIDToText(test)))
        conf = get_activations_single_layer(model,np.array([test]),self.layerName(-1))
        print("current confidence: %s\n"%(conf))
        return conf

    def pre_processing_X(self): 
        self.X_train = sequence.pad_sequences(self.X_train, maxlen=self.max_review_length) 
        self.X_test = sequence.pad_sequences(self.X_test, maxlen=self.max_review_length) 
        
    def pre_processing_x(self,tmp):
        tmp_padded = sequence.pad_sequences([tmp], maxlen=self.max_review_length) 
        #print("%s. Sentiment: %s" % (review,model.predict(np.array([tmp_padded][0]))[0][0]))
        test = np.array([tmp_padded][0])
        #print("input shape: %s"%(str(test.shape)))
        return test
        
    def validateID(self,ids):
        flag = False 
        ids2 = []
        for id in ids: 
            if not (id  in self.id_to_word.keys()): 
                ids2.append(min(self.id_to_word.keys(), key=lambda x:abs(x-id)))
                flag = True
            else: 
                ids2.append(id)
        if flag == True: 
            return validateID(ids2)
        else: return ids


        
    def displayIDRange(self):
        minID = min(self.word_to_id.values())+self.INDEX_FROM
        maxID = max(self.word_to_id.values())+self.INDEX_FROM
        print("ID range: %s--%s"%(minID,maxID))
        
    def fromTextToID(self,review): 
        tmp = []
        for word in review.split(" "):
            tmp.append(self.word_to_id[word])
        return tmp
    
    def fromIDToText(self,ids): 
        tmp = ""
        for id in ids:
            if id > 2: 
                tmp += self.id_to_word[id] + " "
        return tmp
        


'''
units = int(int(model.layers[1].trainable_weights[0].shape[1])/4)
print("No units: ", units)

lstm_layer = model.layers[1] 
print(lstm_layer.get_config())
(W,U,b) = lstm_layer.get_weights()
ns = lstm_layer.trainable_weights
print(W.shape)
print(U.shape)
print(b.shape)

W_i = W[:, :units]
W_f = W[:, units: units * 2]
W_c = W[:, units * 2: units * 3]
W_o = W[:, units * 3:]

print("W_i shape = %s"%(str(W_i.shape)))
print("W_o shape = %s"%(str(W_o.shape)))


U_i = U[:, :units]
U_f = U[:, units: units * 2]
U_c = U[:, units * 2: units * 3]
U_o = U[:, units * 3:]

print("U_i shape=%s"%(str(U_i.shape)))
print("U_o shape=%s"%(str(U_o.shape)))

b_i = b[:units]
b_f = b[units: units * 2]
b_c = b[units * 2: units * 3]
b_o = b[units * 3:]

print("b_i shape=%s"%(str(b_i.shape)))


def step(x, states, W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, b_i, b_f, b_c, b_o):
        assert len(states) == 2
        h_tm1 = states[0]
        c_tm1 = states[1]

        x_i = K.dot(x, W_i) + b_i
        x_f = K.dot(x, W_f) + b_f
        x_c = K.dot(x, W_c) + b_c
        x_o = K.dot(x, W_o) + b_o

        i = K.activations.sigmoid(x_i + K.dot(h_tm1, U_i))
        f = K.activations.sigmoid(x_f + K.dot(h_tm1, U_f))
        c = f * c_tm1 + i * K.activations.tanh(x_c + K.dot(h_tm1, U_c))
        o = K.activations.sigmoid(x_o + K.dot(h_tm1, U_o))
        h = o * K.activations.tanh(c)
        return h, [h, c]
'''
