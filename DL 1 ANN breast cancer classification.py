
# # ANN classification

# In[45]:


import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from keras import optimizers
from sklearn.model_selection import train_test_split
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from sklearn import datasets
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score


# ## Load the data

# In[46]:


# Load the dataset from scikit's data sets
loaddata = datasets.load_breast_cancer()
X, y = loaddata.data, diabetes.target

X = pd.DataFrame(loaddata.data, columns=loaddata.feature_names)
y = pd.DataFrame(loaddata.target)

#preview
X.head()


# In[22]:


#feature scoring
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)


dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features


# In[23]:


counts, bins = np.histogram(y)
plt.hist(bins[:-1], bins, weights=counts)


# In[27]:


# Split the data up in train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)


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
history=model.fit(X_train, y_train,validation_split=0.33,epochs=100, batch_size=10, verbose=1)


# In[38]:


model.summary()


# ## Evaluating the model

# In[39]:


#evaluate the model
score = model.evaluate(X_test, y_test)
print(score)


# In[40]:


#weights
weights=model.layers[0].get_weights()[0]
weights.shape
print("max: ",weights.max() ,"min: ",weights.min())


# In[41]:


#bias
bias=model.layers[0].get_weights()[1]
bias.shape
print("max: ",bias.max() ,"min: ",bias.min())


# In[42]:


#plot the val_accuracy over epochs
print(history.history.keys())
plt.figure(figsize=(15,5))
plt.plot(history.history['val_accuracy'])

plt.ylabel('val_accuracy')
plt.xlabel('epoch')
plt.show()





