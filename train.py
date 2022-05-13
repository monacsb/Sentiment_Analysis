# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:44:24 2022

this train.py file trains the sentiment to determine if the review
is positive or negative

@author: mona

"""

import pandas as pd
from modules import ExploratoryDataAnalysis, ModelCreation
from modules import ModelEvaluation
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import datetime

URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_data.json')
PATH_LOG = os.path.join(os.getcwd(),'Log')
MODEL_PATH = os.path.join(os.getcwd(), 'sentiment.h5')
# %%

#EDA
#Step 1: Import data

df = pd.read_csv(URL)
review = df['review']
sentiment = df['sentiment']

#%% 
#Step 2: data cleaning
#remove tags

eda = ExploratoryDataAnalysis()
review = eda.remove_tags(review) #to remove tags
review = eda.lower_split(review) #to convert to lower case & split

#%%
# Step 3: Features Selection
#Step 4:Data vectorization
review = eda.sentiment_tokenizer(review, TOKENIZER_PATH)
review = eda.sentiment_pad_sequences(review)

#%%
#Step 5: Preprocessing
#one hot coding

one_hot=OneHotEncoder(sparse=False)
sentiment = one_hot.fit_transform(np.expand_dims(sentiment,axis=-1))

#to calculate the num of total
nb_categories = len(np.unique(sentiment))

#x is review (features)
#y is target
#split train test

X_train, X_test, y_train, y_test = train_test_split(review,
                                                    sentiment,
                                                    test_size=0.3,
                                                    random_state=123)

#Expand dimension to fit into model
X_train=np.expand_dims(X_train,axis=-1)
X_test=np.expand_dims(X_test,axis=-1)

#from here you will know that [0,1] is positive, [1,0] is negative
# print(y_train[0])
# print(one_hot.inverse_transform(np.expand_dims(y_train[0],axis=0)))

#model creation
mc = ModelCreation()

num_words = 10000
model = mc.lstm_layer(num_words,nb_categories)

log_dir = os.path.join(PATH_LOG,
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#compile and model fitting
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

hist = model.fit(X_train,y_train,epochs=5,validation_data=(X_test,y_test),
          callbacks=tensorboard_callback)

# %%model evaluation
# preallocation of memory approach

predicted_advanced = np.empty([len(X_test),2]) #the 2 is refering to no of categories

for index, test in enumerate(X_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(test,axis=0))
    
# %%model analysis

y_pred = np.argmax(predicted_advanced,axis=1)
y_true = np.argmax(y_test,axis=1)

me = ModelEvaluation()
me.report_metrics(y_true,y_pred)

# %% model deployment
model.save(MODEL_PATH)

