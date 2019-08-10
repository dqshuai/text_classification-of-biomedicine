"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85% after 2-5 epochs with following settings:
embedding_dim = 50          
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

Differences from original article:
- larger IMDB corpus, longer sentences; sentence length is very important, just like data size
- smaller embedding dimension, 50 instead of 300
- 2 filter sizes instead of original 3
- fewer filters; original work uses 100, experiments show that 3-10 is enough;
- random initialization is no worse than word2vec init on IMDB corpus
- sliding Max Pooling instead of original Global Pooling
"""

import numpy as np
import data_helpers
import data_helpers_elmo
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate

from keras import backend as K
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer
from attention import Attention,Attention_weight
import tensorflow_hub as hub
from keras import regularizers
from Capsule import Capsule
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
import keras
# ---------------------- Parameters section -------------------
#

# Model Hyperparameters
embedding_dim = 1024
filter_sizes = (3,4)
num_filters = 128
dropout_prob = (0.5, 0.8)
hidden_dims = 100

# Training parameters
batch_size = 8
num_epochs = 6



#
# ---------------------- Parameters end -----------------------
def Precision(y_true, y_pred):
    """精确率"""
    tp= K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp= K.sum(K.round(K.clip(y_pred, 0, 1))) # predicted positives
    precision = tp/ (pp+ K.epsilon())
    return precision
    
def Recall(y_true, y_pred):
    """召回率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1))) # possible positives
    recall = tp / (pp + K.epsilon())
    return recall
 
def F1(y_true, y_pred):
    """F1-score"""
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1 
class ELMoEmbedding(Layer):

    def __init__(self, idx2word, output_mode="elmo", trainable=True, **kwargs):
        assert output_mode in ["default", "word_emb", "lstm_outputs1", "lstm_outputs2", "elmo"]
        assert trainable in [True, False]
        self.idx2word = idx2word
        self.output_mode = output_mode
        self.trainable = trainable
        self.max_length = None
        self.word_mapping = None
        self.lookup_table = None
        self.elmo_model = None
        self.embedding = None
        super(ELMoEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.max_length = input_shape[1]
        self.word_mapping = [x[1] for x in sorted(self.idx2word.items(), key=lambda x: x[0])]
        self.lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(self.word_mapping, default_value="<PAD/>")
        sess = tf.Session()
        K.set_session(sess)      
        tf.logging.set_verbosity(tf.logging.ERROR)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        #self.lookup_table.init.run(session=K.get_session())
        self.elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self.trainable)
        super(ELMoEmbedding, self).build(input_shape)

    def call(self, x):
        x = tf.cast(x, dtype=tf.int64)
        sequence_lengths = tf.cast(tf.count_nonzero(x, axis=1), dtype=tf.int32)
        strings = self.lookup_table.lookup(x)
        sess=tf.Session()
        sess.run(tf.tables_initializer())
        inputs = {
            "tokens": strings,
            "sequence_len": sequence_lengths
        }
        return self.elmo_model(inputs, signature="tokens", as_dict=True)[self.output_mode]

    def compute_output_shape(self, input_shape):
        if self.output_mode == "default":
            return (input_shape[0], 1024)
        if self.output_mode == "word_emb":
            return (input_shape[0], self.max_length, 512)
        if self.output_mode == "lstm_outputs1":
            return (input_shape[0], self.max_length, 1024)
        if self.output_mode == "lstm_outputs2":
            return (input_shape[0], self.max_length, 1024)
        if self.output_mode == "elmo":
            return (input_shape[0], self.max_length, 1024)

    def get_config(self):
        config = {
            'idx2word': self.idx2word,
            'output_mode': self.output_mode 
        }
        base_config = super(ELMoEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
"""
def ElmoEmbedding(x):
    
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x[0], tf.string)),
                            "sequence_len":tf.squeeze(tf.cast(x[1], tf.int32))
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]
    
    
    
    return elmo_model(x,
                  signature="default",
                  as_dict=True)["elmo"]
"""
def load_data():
    #x, y,sequence_length= data_helpers_elmo.load_data()
    x, y, vocabulary, vocabulary_inv_list,sequence_length,x_t,y_t = data_helpers.load_data()
    #print(x[0])
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    y = y.argmax(axis=1)
    y_t = y_t.argmax(axis=1)
    #print(y[0])
    # Shuffle data
    np.random.seed(9)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = np.array(x)[shuffle_indices]
    #x = x[shuffle_indices]
    y = y[shuffle_indices]
    
    train_len = int(len(x) * 0.9)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]
    return x_train, y_train, x_test, y_test, vocabulary_inv,sequence_length,x_t,y_t
def load_data_test():
    x, y, vocabulary, vocabulary_inv_list,sequence_length= data_helpers.load_data_test()
    y = y.argmax(axis=1)

    return x, y
# Data Preparation
print("Load data...")
#x_train, y_train, x_test, y_test,sequence_length,x_t,y_t = load_data()
x_train, y_train, x_test, y_test, vocabulary_inv,sequence_length,x_t,y_t= load_data()
#x_t,y_t=load_data_test()

print("load data finished")

# Build model


class Babysiter(keras.callbacks.Callback): 
    def __init__(self):
        super(Babysiter, self).__init__()

    def on_epoch_end(self, batch, logs={}):
        y_pre = model.predict(x_t,batch_size=batch_size)
        y_predict=[]
    
        for i in range(y_pre.shape[0]):
            if(y_pre[i][0]>0.5):
                y_predict.append(1)
            else:
                y_predict.append(0)
        y_predict = np.array(y_predict)
        precision=precision_score(y_t, y_predict)
        recall=recall_score(y_t, y_predict)
        f1=f1_score(y_t, y_predict)
        print("precision：%0.3f"%precision)
        print("recall: %0.3f"%recall)
        print("f1: %0.3f"%f1)
        print("                            ")
"""
sess = tf.Session()
K.set_session(sess)

tf.logging.set_verbosity(tf.logging.ERROR)

#elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
#print("elmo_model initialized finished")
"""
#sess=tf.Session()
#sess.run(tf.tables_initializer())
input_text = Input(shape=(sequence_length,), dtype='int32')
#input_text = Input(shape=(sequence_length,), dtype="string")
input_length = layers.Input(shape=(1,), dtype=tf.int32)
#embeddings = layers.Lambda(ElmoEmbedding, output_shape=(sequence_length,embedding_dim))([input_text,input_length])
#embeddings = layers.Lambda(ElmoEmbedding, output_shape=(sequence_length,embedding_dim))(input_text)
embeddings = ELMoEmbedding(idx2word=vocabulary_inv,output_mode="elmo")(input_text)

z,weight=Attention_weight(8,64)([embeddings, embeddings, embeddings])
#z=embeddings
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         #padding="same",
                         activation="relu",
                         kernel_regularizer=regularizers.l2(0.01),
                         #activity_regularizer=regularizers.l1(0.01),
                         strides=1)(z)
    #conv = Capsule(2, 64, 3, True)(conv)
    conv = MaxPooling1D(pool_size=3)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)

z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(inputs=input_text, outputs=model_output)
#model = Model(inputs=[input_text,input_length], outputs=model_output)
adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(
              loss="binary_crossentropy", 
              optimizer="adam", 
              #optimizer=adam,
              #metrics=[Precision,Recall,F1]
              metrics=[Precision,Recall,F1]
              )
model.summary()


#print(x_train_word_length[0:10])
#print(x_test_word_length[0:10])
filepath="./model_elmo/model_{epoch:02d}.hdf5"
early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1, 
                            save_best_only=False,save_weights_only=False,mode='auto', period=1)

model.fit(
          #[x_train,x_train_word_length], 
          x_train, 
          y_train, batch_size=batch_size, epochs=num_epochs,
          #validation_data=([x_test,x_test_word_length], y_test),
          validation_data=(x_test, y_test),
          verbose=1,
          callbacks=[Babysiter(),checkpoint]
          )

#model.save('ElmoModel.h5')

