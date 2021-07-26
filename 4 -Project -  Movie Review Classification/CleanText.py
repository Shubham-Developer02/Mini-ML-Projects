#!/usr/bin/env python
# coding: utf-8

# # Clean a NLP Pipeline to 'Clean' Reviews Data 
# * Load Input file and read reviews
# * Tokenize
# * Remove Stopwords
# * Perform Stemming
# * Write Clean data to output file

# In[1]:


import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# In[2]:


tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()


# In[3]:


def getCleannedReview(review) :
    review = review.lower()
    review = review.replace("<br /><br />"," ")
    # Tokenize
    tokens = tokenizer.tokenize(review)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    
    cleaned_review = ' '.join(stemmed_tokens)
    
    return cleaned_review


# In[ ]:





# In[4]:


def getStemmedDocument(inputFile,outputFile) :
    output = open(outputFile,'w',encoding='utf8')
    with open(inputFile,encoding='utf8') as f :
        reviews = f.readlines()
        
    for review in reviews :
        cleaned_review = getCleannedReview(review)
        print((cleaned_review),file=output)
        
    output.close()


# In[5]:


inputFile = 'imdb_temp.txt'
outputFile = 'imdb_temp_output.txt'


# In[6]:


getStemmedDocument(inputFile,outputFile)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




