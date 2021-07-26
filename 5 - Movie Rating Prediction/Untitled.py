#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


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


# In[4]:


def getStemmedDocument(reviews) :
    clean_document = []
    for review in reviews :
        cleaned_review = getCleannedReview(review)
        clean_document.append(cleaned_review)
        
    return clean_document


# In[5]:


Dict = {'pos': 1, 'neg': 0}


# In[6]:


train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')


# In[7]:


train_data.columns


# In[9]:


clean_train_data = getStemmedDocument(train_data['review'])


# In[10]:


clean_train_data[0]


# In[12]:


type(clean_train_data)


# In[34]:


train_result = [Dict[i] for i in train_data['label']]


# In[36]:


train_result = np.array(train_result)


# In[37]:


type(train_result)


# In[38]:


train_result.shape


# In[20]:


clean_test_data = getStemmedDocument(test_data['review'])


# In[21]:


len(clean_test_data)


# # Vectorization

# In[24]:


cv = CountVectorizer(ngram_range=(1,2))


# In[41]:


xtrain_vec = cv.fit_transform(clean_train_data)


# In[42]:


type(xtrain_vec)


# In[43]:


xtrain_vec = xtrain_vec.toarray()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




