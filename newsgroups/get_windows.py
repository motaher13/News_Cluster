from fetch_data import fetch_data_groups
import numpy as np
from tqdm import tqdm
import spacy
from gensim import corpora, models
import os

#import sys
#sys.path.append('..')
from utils import preprocess, get_windows

def lda(path):
    
    MIN_COUNTS = 20
    MAX_COUNTS = 1800
    # words with count < MIN_COUNTS
    # and count > MAX_COUNTS
    # will be removed
    
    MIN_LENGTH = 15
    # minimum document length 
    # (number of words)
    # after preprocessing
    
    # half the size of the context around a word
    HALF_WINDOW_SIZE = 5
    # it must be that 2*HALF_WINDOW_SIZE < MIN_LENGTH
    
    
    

    
    
    
    nlp = spacy.load('en')
    
    dataset = fetch_data_groups(data_home=path)
    docs = dataset['data']
    paths=dataset['filenames']
    
    filenames=[path.split("/")[-1] for i, path in enumerate(paths)]
    docs = [(i, doc) for i, doc in enumerate(docs)]
    
    
    
    
     
    
    
    
    encoded_docs, decoder, word_counts = preprocess(
        docs, nlp, MIN_LENGTH, MIN_COUNTS, MAX_COUNTS
    )
    
    
    
    
    
    
    # new ids will be created for the documents.
    # create a way of restoring initial ids:
    doc_decoder = {i: doc_id for i, (doc_id, doc) in enumerate(encoded_docs)}
    filename_decoder={i:filenames[encoded_docs[i][0]] for i in range(len(encoded_docs))}
    
    
    
    
    
    
    data = []
    # new ids are created here
    for index, (_, doc) in enumerate(encoded_docs):
        windows = get_windows(doc, HALF_WINDOW_SIZE)
        # index represents id of a document, 
        # windows is a list of (word, window around this word),
        # where word is in the document
        data += [[index, w[0]] + w[1] for w in windows]
    
    data = np.array(data, dtype='int64')
    
    
    
    
    
    
    
    word_counts = np.array(word_counts)
    unigram_distribution = word_counts/sum(word_counts)
    
    
    
    
    
    
    
    
    vocab_size = len(decoder)
    embedding_dim = 50
    
    # train a skip-gram word2vec model
    texts = [[str(j) for j in doc] for i, doc in encoded_docs]
    model = models.Word2Vec(texts, size=embedding_dim, window=5, workers=4, sg=1, negative=15, iter=70)
    model.init_sims(replace=True)
    
    word_vectors = np.zeros((vocab_size, embedding_dim)).astype('float32')
    for i in decoder:
        word_vectors[i] = model.wv[str(i)]
    	
    	
    	
    	
    	
    
    	
    	
    	
    	
    texts = [[decoder[j] for j in doc] for i, doc in encoded_docs]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    
    
    
    
    
    
    
    n_topics = 2
    lda = models.LdaModel(corpus, alpha=0.9, id2word=dictionary, num_topics=n_topics)
    corpus_lda = lda[corpus]
   
    	
    	
    	
    	
    	
    	
    	
    	
    	
    doc_weights_init = np.zeros((len(corpus_lda), n_topics))
    for i in range(len(corpus_lda)):
        topics = corpus_lda[i]
        for j, prob in topics:
            doc_weights_init[i, j] = prob
    
    		
    		
    		
    		
    		
    		
    		
    		
    print(os.getcwd())		
    np.save('newsgroups/utils/npy/data.npy', data)
    np.save('newsgroups/utils/npy/word_vectors.npy', word_vectors)
    np.save('newsgroups/utils/npy/unigram_distribution.npy', unigram_distribution)
    np.save('newsgroups/utils/npy/decoder.npy', decoder)
    np.save('newsgroups/utils/npy/doc_decoder.npy', doc_decoder)
    np.save('newsgroups/utils/npy/doc_weights_init.npy', doc_weights_init)
    np.save('newsgroups/utils/npy/filename_decoder.npy', filename_decoder)
    
     
    
    
    
    
    
    
    
    lst=[]
    for i, topics in lda.show_topics(n_topics, formatted=False):
        lst.append('topic'+str(i)+':'+' '.join([t for t, _ in topics]))
    	
    return lst


if __name__ == '__main__':
    path="/home/motaher/Desktop/motaher/data"
    lst=lda(path)
    print(lst)