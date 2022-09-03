#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sbn
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# In[7]:


train_ds = pd.read_csv('E:/final rnn/sales_train.csv')
train_ds.head(10)


# In[8]:


test_ds = pd.read_csv('E:/final rnn/test.csv')
test_ds.head(10)


# In[9]:


plt.figure(figsize=(20,5))
ax=sbn.countplot(data=test_ds, x='shop_id')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Frequency of Shop ID in test data")
plt.show()


# In[10]:


monthly_data = train_ds.pivot_table(
    index = ['shop_id','item_id'],
    values = ['item_cnt_day'],
    columns = ['date_block_num'],
    fill_value = 0,
    aggfunc='sum')


# In[11]:


monthly_data.reset_index(inplace = True)
monthly_data.head()


# In[12]:


train_data = monthly_data.drop(columns= ['shop_id','item_id'], level=0)


# In[13]:


train_data.fillna(0,inplace = True)
train_data.head()


# In[14]:


x_train = np.expand_dims(train_data.values[:,:-1],axis = 2)
y_train = train_data.values[:,-1:]


# In[15]:


test_rows = monthly_data.merge(
    test_ds,
    on = ['item_id','shop_id'],
    how = 'right')


# In[16]:


x_test = test_rows.drop(test_rows.columns[:5], axis=1).drop('ID', axis=1)


# In[17]:


x_test.fillna(0,inplace = True)


# In[18]:


x_test = np.expand_dims(x_test,axis = 2)


# In[24]:


model = tf.keras.models.Sequential()    
model.add(LSTM(64, input_shape=(33, 1), return_sequences=False))
model.add(Dense(1))
    
model.compile(
    loss = 'mse',
    optimizer = 'adam', 
    metrics = ['mean_squared_error']        
)


# In[20]:


history = model.fit(
    x_train, 
    y_train, 
    epochs=10, 
    batch_size=4096,
    verbose=1, 
    shuffle=True,
    validation_split=0.4)





# In[25]:


plt.figure(figsize=(20,5))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='best')
plt.show()





