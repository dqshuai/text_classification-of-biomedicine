import numpy as np
import re
import itertools
from collections import Counter
import json
"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string):
    #string = re.sub(r"[^A-Za-z0-9(),!?\-\'\`]", " ", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
#    string = re.sub(r",", " , ", string)
#    string = re.sub(r"!", " ! ", string)
#    string = re.sub(r"\(", " \( ", string)
#    string = re.sub(r"\)", " \) ", string)
#    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels_test():
    with open("./data/PMtask_Triage_TestSet.json",'r',encoding='utf-8') as load_f:
        load_dict=json.load(load_f)
    x=[]
    y=[]
    
    for i in range(len(load_dict["documents"])):
#        x.append(load_dict["documents"][i]["passages"][0]["text"])
        if(len(load_dict["documents"][i]["passages"])==2):            
            #x.append(load_dict["documents"][i]["passages"][0]["text"]+" "+load_dict["documents"][i]["passages"][1]["text"])
            x.append(load_dict["documents"][i]["passages"][1]["text"])
#        else:
#            x.append(load_dict["documents"][i]["passages"][0]["text"])
            if(load_dict["documents"][i]["infons"]["relevant"]=="yes"):
                #count_1=count_1+1
                y.append([1,0])
            elif(load_dict["documents"][i]["infons"]["relevant"]=="no"):
                #count_2=count_2+1
                y.append([0,1])
    
    x=[s.strip() for s in x]
    x_text=[clean_str(sent) for sent in x]
    x_text=[s.split(" ") for s in x_text]
    #y=np.array(y)
    #print(count_1,count_2)
    return [x_text,y]
def load_data_and_labels():
    with open("./data/PMtask_Triage_TrainingSet.json",'r',encoding='utf-8') as load_f:
        load_dict=json.load(load_f)
    x=[]
    y=[]
    
    for i in range(len(load_dict["documents"])):
#        x.append(load_dict["documents"][i]["passages"][0]["text"])
        if(len(load_dict["documents"][i]["passages"])==2):            
            #x.append(load_dict["documents"][i]["passages"][0]["text"]+" "+load_dict["documents"][i]["passages"][1]["text"])
            x.append(load_dict["documents"][i]["passages"][1]["text"])
#        else:
#            x.append(load_dict["documents"][i]["passages"][0]["text"])
            if(load_dict["documents"][i]["infons"]["relevant"]=="yes"):
                #count_1=count_1+1
                y.append([1,0])
            elif(load_dict["documents"][i]["infons"]["relevant"]=="no"):
                #count_2=count_2+1
                y.append([0,1])
    
    """
    for i in range(len(load_dict["documents"])):
        if(len(load_dict["documents"][i]["passages"])==2):
            str=load_dict["documents"][i]["passages"][1]["text"].split(". ")
            str_temp=""
            for j in range(len(str)):
                str_temp=str_temp+" "+str[j]
                if((j+1)%3==0):#5句话一划分
                    x.append(str_temp)
                    str_temp=""
                    if(load_dict["documents"][i]["infons"]["relevant"]=="yes"):
                        y.append([1,0])
                    elif(load_dict["documents"][i]["infons"]["relevant"]=="no"):
                        y.append([0,1])
            if(str_temp!=""):
                x.append(str_temp)
                str_temp=""
                if(load_dict["documents"][i]["infons"]["relevant"]=="yes"):
                    y.append([1,0])
                elif(load_dict["documents"][i]["infons"]["relevant"]=="no"):
                    y.append([0,1])
    """

    """
    for i in range(len(load_dict["documents"])):
#        x.append(load_dict["documents"][i]["passages"][0]["text"])
        if(len(load_dict["documents"][i]["passages"])==2):
            x.append(load_dict["documents"][i]["passages"][1]["text"])
            if(load_dict["documents"][i]["infons"]["relevant"]=="yes"):
                #count_1=count_1+1
                y.append([1,0])
            elif(load_dict["documents"][i]["infons"]["relevant"]=="no"):
                #count_2=count_2+1
                y.append([0,1])
    """        
    """
    for i in range(len(load_dict["documents"])):
        x.append(load_dict["documents"][i]["passages"][0]["text"])
        if(load_dict["documents"][i]["infons"]["relevant"]=="yes"):
            #count_1=count_1+1
            y.append([1,0])
        elif(load_dict["documents"][i]["infons"]["relevant"]=="no"):
            #count_2=count_2+1
            y.append([0,1])
    """
    x=[s.strip() for s in x]
    x_text=[clean_str(sent) for sent in x]
    x_text=[s.split(" ") for s in x_text]
    #print(len(x_text))
    #y=np.array(y)
    #print(count_1,count_2)
    return [x_text,y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        new_sentence = new_sentence[0:450]
        padded_sentences.append(new_sentence)
    #return padded_sentences,sequence_length
    return padded_sentences,450


def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded,sequence_length = pad_sentences(sentences)
    sentences_test, labels_test = load_data_and_labels_test()
    sentences_padded_test,_ = pad_sentences(sentences_test)
    sentences_padded_total=sentences_padded+sentences_padded_test
    vocabulary, vocabulary_inv = build_vocab(sentences_padded_total)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    x_t, y_t = build_input_data(sentences_padded_test, labels_test, vocabulary)
    return [x, y, vocabulary, vocabulary_inv,sequence_length,x_t,y_t]


def load_data_test():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels_test()
    sentences_padded,sequence_length = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv,sequence_length]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
if __name__=='__main__':
    x,y,_,_,_=load_data()
    #print(x[0])
