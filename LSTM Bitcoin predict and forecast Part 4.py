#!/usr/bin/env python
# coding: utf-8

# #  LSTM stock price prediction

# In[1]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential
from keras.layers import LSTM, Dense,Flatten, Dropout, Activation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler 
import matplotlib.cbook as cbook
from keras import optimizers
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas.tseries.offsets import DateOffset
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import warnings
warnings.filterwarnings("ignore")
import chart_studio.plotly.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go
pyoff.init_notebook_mode(connected=True)


# In[3]:


def parser(x):
    return pd.datetime.strptime(x, '%Y-%m-%d')
df = pd.read_csv('C:/Users/Harry/Desktop/Analytics/BTC stock price.csv', parse_dates=[0], index_col=0, date_parser=parser)
df = df.dropna()
Open = df[["Open"]]
Open


# In[4]:



scaler = MinMaxScaler()
scaler.fit(Open)
train = scaler.transform(Open)
train.shape


# In[106]:


n_input = 30
n_features = 1

#generate time series sequences for the forecast 
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=8)
generator


# ## Training the model 

# In[107]:


model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer="nadam", loss="mse",metrics=['accuracy'])
history = model.fit_generator(generator,epochs=10)


# ## Evaluating the model and predict

# In[99]:


#evaluate the model
score = model.evaluate(generator)
print(score)


# In[100]:


#plot loss over number of epochs
plt.figure(figsize=(15,5))
plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[108]:


#predict the data
pred_train=model.predict(generator)
pred_train=scaler.inverse_transform(pred_train)
pred_train=pred_train.reshape(-1)


# In[109]:


pred_list = []

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)


# In[110]:


add_dates = [Open.index[-1] + DateOffset(days=x) for x in range(0,31) ]
future_dates = pd.DataFrame(index=add_dates[1:],columns=Open.columns)


# In[111]:


#calculate the forecast
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=future_dates[-n_input:].index, columns=['Forecast'])

df_proj = pd.concat([Open,df_predict], axis=1)

df_proj.tail(31)


# ## Ploting the result

# In[112]:


plot_data = [
    go.Scatter(
        x=df_proj.index,
        y=df_proj['Open'],
        name='Actual'
    ),
    go.Scatter(
        x=df_proj.index,
        y=df_proj['Forecast'],
        name='Forecast'
    ),
      go.Scatter(
        x=df_proj.index,
        y=pred_train,
        name='Prediction'
    )
]
plot_layout = go.Layout(
        title='Bitcoin stock price prediction'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

