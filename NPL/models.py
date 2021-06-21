 
# from flask import Flask 
# from flask_ngrok import run_with_ngrok
# from flask import request, jsonify 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.initializers import Constant
from keras.layers.embeddings import Embedding
# مثال عن ال word2vec
### تضمني جميع المكاتب المحتاجة لتضمين الكلمات
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.initializers import Constant
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.initializers import Constant
from keras.layers.embeddings import Embedding
## https://arabicprogrammer.com/article/871884237/
## في هذا الرابط شرح لكتبة  nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.initializers import Constant
from keras.layers.embeddings import Embedding
import numpy as np
import pandas as pd
import os






def convert_to_lower(text):
  return text.lower()
####################################################################################
def remove_numbers(text):
  text = re.sub(r'\d+' , '', text)
  return text
####################################################################################
def remove_punctuation(text):
     punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
     no_punct = ""
     for char in text:
        if char not in punctuations:
            no_punct = no_punct + char
     return no_punct
####################################################################################
def remove_white_space(text):
  text = text.strip()
  return text
####################################################################################
## تقطيع الجمل الى مجموعة من الكلمات 
def toknizing(text):
  ## إزالة كلمات ال stopwords 
  ## https://ar.wikipedia.org/wiki/%D8%A7%D8%B3%D8%AA%D8%A8%D8%B9%D8%A7%D8%AF_%D8%A7%D9%84%D9%83%D9%84%D9%85%D8%A7%D8%AA_%D8%A7%D9%84%D8%B4%D8%A7%D8%A6%D8%B9%D8%A9
  ## في هذا الرابط شرح عن الكلمات الشائعة 
  stop_words = set(stopwords.words('english'))
  tokens = word_tokenize(text)
  ## Remove Stopwords from tokens
  result = [i for i in tokens if not i in stop_words]
  return result
###################
## تابع لمعالجة كلمات الدخل والنصوص
def pre_processing(text):
  text = convert_to_lower(text)
  text = remove_punctuation(text)
  text = remove_white_space(text)
  text = remove_numbers(text)
  return text
  
## تابع لمعالجة كلمات الدخل والنصوص
def pre_processing_with_tokinizing(text):
  text = convert_to_lower(text)
  text = remove_punctuation(text)
  text = remove_white_space(text)
  text = remove_numbers(text)
  #text = toknizing(text)
  return text 

















print(str(os.getcwd()))
print(os.listdir(str(os.getcwd())))
csv_file= pd.read_csv('csv_tweet.csv') 
dataframe = csv_file[['id' , 'category' , 'tweet' ]]
dataframe = dataframe.replace(2,1)
print("dataframe\n",dataframe)
normal_tweet = 0
suicide_tweet = 0
for i in dataframe['category'].values:
  if i == 0 :
    normal_tweet = normal_tweet + 1

  else : 
    suicide_tweet = suicide_tweet + 1

print(normal_tweet)
print(suicide_tweet)
# import numpy as np
nltk.download("stopwords")
nltk.download('punkt')
# ندرب word2vec 
## تضمين الكلمات 
all_data = [pre_processing(i) for i in dataframe['tweet']]
word2vec_model = gensim.models.Word2Vec(sentences= all_data , vector_size = 100 , window = 5 , workers =4 , min_count =1)
## train 100 عينة 
## test 50 عينة 
data= [pre_processing_with_tokinizing(i) for i in dataframe['tweet']]
x_train , x_test , y_train , y_test = train_test_split(data, np.array(dataframe['category']) )
tokenizner = Tokenizer()
tokenizner.fit_on_texts(x_train + x_test)
max_length = max([len(s.split()) for s in data])
vocab_size = len(tokenizner.word_index)+1
x_train_token = tokenizner.texts_to_sequences(x_train)
x_test_token = tokenizner.texts_to_sequences(x_test)
embedding_size = 100
embedding_matrix = np.zeros((vocab_size , embedding_size))
for word ,i in tokenizner.word_index.items():
  if (i > vocab_size):
    break;
  try:
    embedding_matrix[i] = word2vec_model.wv[word]
  except: 
    continue;
  
x_train_pad = pad_sequences(x_train_token , maxlen= max_length , padding = 'post')
x_test_pad = pad_sequences(x_test_token , maxlen= max_length , padding = 'post')



model = Sequential()
model.add(Embedding(vocab_size , embedding_size , embeddings_initializer = Constant(embedding_matrix)  , input_length = max_length , trainable = True))
model.add(LSTM(4))
model.add(Dense(1 , activation= 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])
history = model.fit(x_train_pad, y_train, validation_data=(x_test_pad , y_test) , epochs=20)


import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


test_sample1 = "Today I just got out of the hospital for an attempted suicide I feel better than before although my mother  think her partner is hurting me ashamed to say it in my personal account,  why I write it in , thanks for asking,  good to know that"
test_sample2 = "survived two suicide attempts and after seeking help she turned her life around "
test_sample3 =  "I On My Block\\nBig Mouth\\nThe Flash\\nGreys Anatomy \\nSex Education \\nBright \\n\\nTW, talk of suicide and self harm\\/\\/  Dead by the end of the week (or your money back) - it\\u2019s kinda dumb but it\\u2019s an ok dumb movie"
test_sample4 = "i will not weak up tommorow"
test_sample5 = "to day is my birth_day"
test_sample6 = "we lost 3 Marines today.\\nTwo veterans committed suicide at VA in GA. \\nWe have so many Veterans in need"
test_sample7 = 'There is a light at the end of the tunnel . but the tunnel is full of problems . shootings , rape , and suicide along '
test_samples = [test_sample1 ,test_sample2 ,test_sample3 ,test_sample4 ,test_sample5 ,test_sample6 ,test_sample7 ]
processed_test_data = [pre_processing(i) for i in test_samples]

test_samples_tokens= tokenizner.texts_to_sequences(processed_test_data)
print(max_length)
test_samples_tokens_pad = pad_sequences(test_samples_tokens,maxlen= max_length)

prediction = model.predict(x=test_samples_tokens_pad)
prediction = prediction.round()
for i in prediction:
  if i == 1:
    print('suicide')
  else :
    print('normal')













def predict(input_text):
    print(max_length)
    test_sample= tokenizner.texts_to_sequences(pre_processing(input_text))
    test_samples_tokens_pad = pad_sequences(test_sample ,maxlen= max_length)
    prediction = model.predict(x=test_samples_tokens_pad)
    print(prediction)
    if prediction[0].round()==1:
      return  'suicide'
    else:
      return  'normal'
