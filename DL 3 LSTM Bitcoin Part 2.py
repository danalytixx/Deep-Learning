#!/usr/bin/env python
# coding: utf-8

# # LSTM stock price prediction

# In[2]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential
from keras.layers import LSTM, Dense,Flatten, Dropout, Activation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler 
import matplotlib.cbook as cbook
from keras import optimizers
from sklearn.model_selection import train_test_split


# ## Load the data

# In[3]:


df = pd.read_csv("C:/Users/Harry/Desktop/Analytics/BTC stock price.csv")
df = df.dropna()

changes = df[["Open", "Volume"]]
changes


# In[4]:


#Min-Max scaling - standardization (normalization)

scaler= MinMaxScaler(feature_range = (0, 1))

# To scale data 
stand=scaler.fit_transform(changes) 

df_stand=pd.DataFrame(stand, columns=["Open", "Volume"])

df_stand.head()


# In[5]:


# using 20 days as a timeframe  
X = []
y = []

changes=df_stand.values
for i in range(0, len(changes) - 20):
    y.append(changes[i,0])
    X.append(np.array(changes[i+1:i+21][::-1]))
    
X = np.array(X).reshape(-1, 20, 2)
y = np.array(y)


X.shape


# In[20]:


# Split the data up in train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ## Training the model 

# In[27]:


# Create NN
model = Sequential()
#LSTM layer
model.add(LSTM(64, activation='relu',input_shape = (20, 2)))
model.add(Dense(32))
#output layer
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse",metrics=['accuracy'])
model.fit(X_train, y_train ,batch_size=32, epochs=50)


# ## Evaluating the model and predict

# In[42]:


#evaluate the model
score = model.evaluate(X_test, y_test)
print(score)


# In[43]:


#calculate the prediction
predictions = model.predict(X)
predictions.shape


# In[44]:


# inverse standardization (normalization) scale back to original
predictions_st = predictions*( (df["Open"].max(axis=0) - df["Open"].min(axis=0))) +df["Open"].min(axis=0) 
predictions_st=np.append(predictions_st, np.zeros(20))
predictions_st=predictions_st.reshape(-1)
predictions_st.shape


# ## Ploting the result

# In[45]:


plt.figure(figsize=(30,10))
dates = np.array(df["Date"]).astype(np.datetime64)

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')

plt.plot(dates[:-20], df["Open"][:-20], label="Open")
plt.plot(dates[:-20], predictions_st[:-20], label="Open (predicted)")

plt.title(' Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[47]:


plt.figure(figsize=(30,10))

# format the ticks
ax = plt.gca()
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')

dates = np.array(df["Date"]).astype(np.datetime64)

plt.plot(dates[-100:-20], df["Open"][-100:-20], label="Open")
plt.plot(dates[-100:-20], predictions_st[-100:-20], label="Open (predicted)")


plt.title(' Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[ ]:




