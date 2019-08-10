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
import keras
import numpy as np
import data_helpers_att
from w2v import train_word2vec
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D
from keras.layers.merge import Concatenate
from keras import backend as K
from attention import Position_Embedding,Position_Embedding_word, Attention,Attention_word,Attention_weight,Attention_word_weight
from Capsule import Capsule
from keras import regularizers
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score
from keras.callbacks import ModelCheckpoint
import tensorflow as tf




# ---------------------- Parameters section -------------------
#
# Model Hyperparameters
embedding_dim = 200

# Training parameters
batch_size = 64
num_epochs = 50


# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 5








#能进行调整的参数
filter_sizes = (2,3)
num_filters = 128
dropout_prob = (0.5, 0.5)
hidden_dims = 100
num_sentence=10#每个文档有几个句子
length_sentence=30#每个句子有几个单词
attenion_1_head=8#attention第一层的头数
attenion_1_dim=32#attention第一层的维度
attenion_2_head=8#attention第二层的头数
attenion_2_dim=32#attention第二层的维度
capsule_dim=32#胶囊网络的维度
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

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)





def load_data():
    x, y, vocabulary, vocabulary_inv_list,x_t,y_t=data_helpers_att.load_data(num_sentence,length_sentence)
    #x, y, vocabulary, vocabulary_inv_list,sequence_length = data_helpers.load_data()
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    y_t = y_t.argmax(axis=1)
    y = y.argmax(axis=1)

    # Shuffle data
    np.random.seed(9)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * 0.9)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]
    
    return x_train, y_train, x_test, y_test, vocabulary_inv,x_t,y_t




def load_data_test():
    x, y, vocabulary, vocabulary_inv_list = data_helpers_att.load_data_test(num_sentence,length_sentence)
    #x, y, vocabulary, vocabulary_inv_list,sequence_length = data_helpers.load_data_test()
    y = y.argmax(axis=1)

    return x, y
    #return x_train, y_train, x_test, y_test, vocabulary_inv,sequence_length
# Data Preparation
print("Load data...")
#x_train, y_train, x_test, y_test = load_data()

#x_train, y_train, x_test, y_test, vocabulary_inv,sequence_length = load_data()

x_train, y_train, x_test, y_test, vocabulary_inv,x_t,y_t = load_data()
#x_t, y_t=load_data_test()
#x_train=x_t
#y_train=y_t
embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                   min_word_count=min_word_count, context=context)
#print(embedding_weights[0])




#print(x_train[0])
#print(x_t[0])
x_train = np.stack([[np.stack([embedding_weights[word] for word in sentence]) for sentence in document] for document in x_train])
x_test = np.stack([[np.stack([embedding_weights[word] for word in sentence]) for sentence in document] for document in x_test])
x_t = np.stack([[np.stack([embedding_weights[word] for word in sentence]) for sentence in document] for document in x_t])

#print(x_train[0])
#print(x_t[0])

#print(y_t[0:10])

#x_train=np.vstack((x_train,x_t))
#y_train=np.concatenate((y_train,y_t),axis=0)

#x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
#x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
#x_t=np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_t])
#print(np.shape(x_train))

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
        auc=roc_auc_score(y_t, y_predict)
        print("precision：%0.3f"%precision)
        print("recall: %0.3f"%recall)
        print("f1: %0.3f"%f1)
        print("auc: %0.3f"%auc)
        print("                            ")


# Build model



input_shape = (num_sentence,length_sentence, embedding_dim)#56 128
#input_shape = (sequence_length, embedding_dim)
model_input = Input(shape=input_shape)
embeddings = model_input
embeddings = Dropout(dropout_prob[0])(embeddings)

#z=Attention_word_weight(8, 64)([embeddings, embeddings, embeddings])
#z=Attention_weight(8,128)([z,z,z])
#z=embeddings
z=Attention_word(attenion_1_head, attenion_1_dim)([embeddings, embeddings, embeddings])
z=Attention(attenion_2_head, attenion_2_dim)([z,z,z])
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         kernel_regularizer=regularizers.l2(0.01),
                         strides=1)(z)
    conv = Capsule(2, capsule_dim, 3, True)(conv)
    
    #conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)

z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model.compile(loss="binary_crossentropy", 
              optimizer="adam", 
              #optimizer=adam,
              metrics=[Precision,Recall,F1,auc]
              )
model.summary()
filepath="./model/model_{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, 
                            save_best_only=False,save_weights_only=False,mode='auto', period=1)
# Train the model


model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          #validation_data=(x_test, y_test), 
          validation_data=(x_test, y_test), 
          verbose=1,
          callbacks=[Babysiter()]
          )

#model.save("model.h5")
"""
for i in range(1,20):
    if(i<10):  
        file="./model/model_"+"0"+str(i)+".hdf5"
    else:
        file="./model/model_"+str(i)+".hdf5"
    model.load_weights(file)
    y_pre = model.predict(x_t,batch_size=batch_size)
    y_predict=[]
    
    for i in range(y_pre.shape[0]):
        if(y_pre[i][0]>0.5):
            y_predict.append(1)
        else:
            y_predict.append(0)
    y_predict = np.array(y_predict)
    #print(y_predict[0:10])
    #print(y_test[0:10])
    #accuracy = accuracy_score(y_t, y_predict)
    precision=precision_score(y_t, y_predict)
    recall=recall_score(y_t, y_predict)
    f1=f1_score(y_t, y_predict)
    print("precision：%0.3f"%precision)
    print("recall: %0.3f"%recall)
    print("f1: %0.3f"%f1)
    print("                            ")
"""
#pred_test = model.predict(x_test,batch_size=batch_size)
#print(pred_test[0])


"""
model.load_weights("./model/model_0.723.hdf5")
y_pre = model.predict(x_t,batch_size=batch_size)
y_predict=[]

for i in range(y_pre.shape[0]):
    if(y_pre[i][0]>0.5):
        y_predict.append(1)
    else:
        y_predict.append(0)
y_t=list(y_t)
#print(y_t[0])
t_t=0
t_f=0
f_t=0
f_f=0
for i in range(len(y_t)):
    if(y_t[i]==1 and y_predict[i]==1):
        t_t=t_t+1
    if(y_t[i]==1 and y_predict[i]==0):
        t_f=t_f+1
    if(y_t[i]==0 and y_predict[i]==1):
        f_t=f_t+1
    if(y_t[i]==0 and y_predict[i]==0):
        f_f=f_f+1
print("混淆矩阵：")
print("Actual                       Predicted")
print("                             true             false")
print("true                        ",t_t,"            ",t_f)
print("false                       ",f_t,"            ",f_f)

y_predict = np.array(y_predict)
#print(y_predict[0:10])
#print(y_test[0:10])
#accuracy = accuracy_score(y_t, y_predict)

precision=precision_score(y_t, y_predict)
recall=recall_score(y_t, y_predict)
f1=f1_score(y_t, y_predict)
print("precision：%0.3f"%precision)
print("recall: %0.3f"%recall)
print("f1: %0.3f"%f1)
"""
