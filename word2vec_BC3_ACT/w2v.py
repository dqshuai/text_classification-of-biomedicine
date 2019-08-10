from __future__ import print_function
from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np
import gensim


def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.
   
    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {int: str}
    num_features    # Word vector dimensionality                      
    min_word_count  # Minimum word count                        
    context         # Context window size 
    """
    """
    model_dir = 'models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        # Set values for various parameters
        num_workers = 2  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model
        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        #sentences = [" ".join(s) for s in sentences]
        #print(sentences[0])
        
        #embed = hub.Module("https://tfhub.dev/google/Wiki-words-250/1")
        #embedding_model = embed(sentences)
        #print(embedding_model[0])

        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            size=num_features, min_count=min_word_count,
                                            #sg=1,
                                            window=context, sample=downsampling)

        # If we don't plan to train the model any further, calling 
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)
        
        # Saving the model for later use. You can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)


    """
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format('./vector/bio_nlp_vec/PubMed-shuffle-win-2.bin', binary=True)
    # add unknown words
    embedding_weights = {key: embedding_model[word] if word in embedding_model else
                              np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in vocabulary_inv.items()}
    
    
    
    #for key in embedding_model.wv.similar_by_word('pitx2',topn=10):
        #print(key[0],key[1])
    #print(embedding_model['pitx2'])
    return embedding_weights
    
if __name__ == '__main__':
    import data_helpers

    print("Loading data...")
    x, _, _, vocabulary_inv_list,sequence_length= data_helpers.load_data()
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    #for i in range(100):
    #    print(vocabulary_inv[i])
    
    
    
    w = train_word2vec(x, vocabulary_inv,num_features=128,min_word_count=1, context=5)
    
    
    print(len(w[0]))
    #train_word2vec(x, vocabulary_inv)
    print(len(w))
    #print(sequence_length)
