#!/usr/bin/env python
# coding: utf-8

# # ANN classification

# In[9]:


import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.utils import resample
from keras import optimizers
from sklearn.model_selection import train_test_split
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from sklearn import datasets
import imblearn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the data

# In[2]:


#Importing traing data
dataset = pd.read_csv('C:/Users/Harry/Desktop/Analytics/creditcard.csv')
X = dataset.iloc[:, 0:30]
y = dataset.iloc[:, 30:31]


#preview
X.head()


# In[14]:


# Split the data up in train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ## Oversampling imbalanced data 

# In[3]:


counts, bins = np.histogram(y)
plt.hist(bins[:-1], bins, weights=counts)


# In[32]:


# transform the dataset with Synthetic Minority Oversampling Technique
oversample = SMOTE(sampling_strategy=0.8)
Xo, yo = oversample.fit_resample(X_train, y_train)


# In[33]:


counts, bins = np.histogram(yo)
plt.hist(bins[:-1], bins, weights=counts)


# ## Training the model 

# In[36]:


# Initialize the NN 
model = Sequential()

# Add input layer 
model.add(Dense(64,  input_shape=(30,), activation='sigmoid'))

# Add output layer 
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])



#train the model
history=model.fit(Xo, yo,validation_split=0.4,epochs=10, batch_size=100, verbose=1)


# In[12]:


model.summary()


# ## Evaluating the model

# In[37]:


#evaluate the model
score = model.evaluate(X_test, y_test)
print(score)


# In[38]:


#weights
weights=model.layers[0].get_weights()[0]
weights.shape
print("max: ",weights.max() ,"min: ",weights.min())


# In[39]:


#bias
bias=model.layers[0].get_weights()[1]
bias.shape
print("max: ",bias.max() ,"min: ",bias.min())


# In[ ]:




