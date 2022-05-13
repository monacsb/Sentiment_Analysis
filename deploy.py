# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:20:49 2022

@author: User
"""

from tensorflow.keras.models import load_model
import os
import json
from modules import ExploratoryDataAnalysis
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np

MODEL_PATH = os.path.join(os.getcwd(), 'sentiment.h5')
JSON_PATH = os.path.join(os.getcwd(),'tokenizer_data.json')
# %%


sentiment_classifier = load_model(MODEL_PATH)
sentiment_classifier.summary()

# %% tokenizer loading
JSON_PATH = os.path.join(os.getcwd(),'tokenizer_data.json')
with open(JSON_PATH, 'r') as json_file:
    token = json.load(json_file)

#%% EDA

# Step1:Load data

# new_review = ['<br \> I think the first half of the movie is interesting but\
#     the second half of the movie is boring. I have wasted my time.<br\>']
    
new_review = [input('Review about the movie: ')]

# Step2:to clean the data
eda = ExploratoryDataAnalysis()
removed_tags = eda.remove_tags(new_review)
cleaned_input = eda.lower_split(removed_tags)

# step 3:features selection
# step4: data preprocessing
# to vectorize the new new review
loaded_tokenizer = tokenizer_from_json(token)

# to vectorize the review into integers
new_review = loaded_tokenizer.texts_to_sequences(new_review)
new_review = eda.sentiment_pad_sequences(new_review)
# %%model prediction
outcome = sentiment_classifier.predict(np.expand_dims(new_review,axis=-1))
# positive = [0,1]
# negative = [1,0]
sentiment_dict = {1:'positive', 0:'negative'}
print('This review is ' + sentiment_dict[np.argmax(outcome)])

