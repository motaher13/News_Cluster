import numpy as np
from tqdm import tqdm
import spacy
from gensim import corpora, models

import torch
from torch.autograd import Variable
import torch.optim as optim
import math

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import json



def lda2vec():

    data = np.load('newsgroups/utils/npy1/data.npy')
    unigram_distribution = np.load('newsgroups/utils/npy1/unigram_distribution.npy')
    word_vectors = np.load('newsgroups/utils/npy1/word_vectors.npy')
    doc_weights_init = np.load('newsgroups/utils/npy1/doc_weights_init.npy')







    doc_weights_init = np.log(doc_weights_init + 1e-4)

    # make distribution softer
    temperature = 7.0
    doc_weights_init /= temperature












    #from utils import train
    #train(
    #    data, unigram_distribution, word_vectors,
    #    doc_weights_init, n_topics=2,
    #    batch_size=1024*7, n_epochs=10,
    #    lambda_const=500.0, num_sampled=15,
    #    topics_weight_decay=1e-2,
    #    topics_lr=1e-3, doc_weights_lr=1e-3, word_vecs_lr=1e-3,
    #    save_every=5, grad_clip=5.0
    #)







    def softmax(x):
        # x has shape [batch_size, n_classes]
        e = np.exp(x)
        n = np.sum(e, 1, keepdims=True)
        return e/n









    from fetch_data import fetch_data_groups
    dataset = fetch_data_groups(data_home='/home/motaher/Desktop/motaher/data')
    docs = dataset['data']
    paths=dataset['filenames']







    # store each document with an initial id
    docs = [(i, doc) for i, doc in enumerate(docs)]

    # "integer -> word" decoder 
    decoder = np.load('newsgroups/utils/npy1/decoder.npy')[()]

    # for restoring document ids, "id used while training -> initial id"
    doc_decoder = np.load('newsgroups/utils/npy1/doc_decoder.npy')[()]
    filename_decoder = np.load('newsgroups/utils/npy1/filename_decoder.npy')[()]






    state = torch.load('newsgroups/model_state.pytorch', map_location=lambda storage, loc: storage)
    n_topics = 2

    doc_weights = state['doc_weights.weight'].cpu().clone().numpy()
    topic_vectors = state['topics.topic_vectors'].cpu().clone().numpy()
    resulted_word_vectors = state['neg.embedding.weight'].cpu().clone().numpy()

    # distribution over the topics for each document
    topic_dist = softmax(doc_weights)

    # vector representation of the documents
    doc_vecs = np.matmul(topic_dist, topic_vectors)










    similarity = np.matmul(topic_vectors, resulted_word_vectors.T)
    most_similar = similarity.argsort(axis=1)[:, -10:]

    #lst=[]
    #for j in range(n_topics):
    #    topic_words = ' '.join([decoder[i] for i in reversed(most_similar[j])])
    #    lst.append('topic'+str( j + 1)+ ':'+ topic_words)
        
    #return lst
    
    doc_weights_init = np.load('newsgroups/utils/npy1/doc_weights_init.npy')
    
    
    length=len(topic_dist)
    output=[]
    for i in range(length):
        item={}
        item['docname']=filename_decoder[i]
        item['doc']=[doc for j, doc in docs if j == doc_decoder[i]][0]
        item['dist_over_topics']={}
        s = ''
        n=0;
        for j, p in enumerate(topic_dist[i], 1):
           if p>0.001:
                n=n+1
                item['dist_over_topics'][str(j)]="%.9f" % p

        item['topics']={}
        for j in reversed(topic_dist[i].argsort()[-n:]):
           topic_words = ' '.join([decoder[i] for i in reversed(most_similar[j])])
           item['topics'][str(j+1)]=topic_words
        
         
        
        output.append(item)
        
    #out=json.dumps(output,ensure_ascii=False)
    return output







    #i = 3 # document id

    #print('DOCUMENT:')
    #print([doc for j, doc in docs if j == doc_decoder[i]][0], '\n')

    #print('DISTRIBUTION OVER TOPICS:')
    #s = ''
    #n=0;
    #for j, p in enumerate(topic_dist[i], 1):
    #    if p>0.0009:
    #        n=n+1
    #        s += '{0}:{1:.3f}  '.format(j, p)
    #        if j%6 == 0:
    #            s += '\n'
    #print(s)

    #print('\nTOP TOPICS:')
    #for j in reversed(topic_dist[i].argsort()[-n:]):
    #    topic_words = ' '.join([decoder[i] for i in reversed(most_similar[j])])
    #    print('topic', j + 1, ':', topic_words)
    
    
    
#if __name__ == '__main__':
#    lst=lda2vec()
#    print(lst)
    
    