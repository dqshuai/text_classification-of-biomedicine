import numpy as np
import re

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
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
    x=[s.strip() for s in x]
    x_text=[clean_str(sent) for sent in x]
    x_text=[s.split(" ") for s in x_text]
    y=np.array(y)
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        #str=""
        #for j in range(len(sentence)):
        #    str=str+sentence[j]+" "
        #padded_sentences.append(str)
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
        
    #x=[]
    #for i in range(len(padded_sentences)):
        #str=tuple(padded_sentences[i])
    #    str=' '.join(padded_sentences[i])
    #    x.append(str)
    return padded_sentences,sequence_length


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded,sentence_length = pad_sentences(sentences)
    return[sentences_padded,labels,sentence_length]


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

#x,y=load_data()
#print(x[0],y[0])