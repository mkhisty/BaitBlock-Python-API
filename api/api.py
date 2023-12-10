from keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from flask import Flask, jsonify, request
import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.text import Tokenizer
import pickle
from tensorflow import keras
import os
import base64

print(os.listdir())
app = Flask(__name__)
from keras.models import Sequential
from keras.layers import  Dense,Embedding

model = Sequential()
model.add(Embedding(10000,32))
model.add(LSTM(32, activation='sigmoid' ))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss=['binary_crossentropy'] , optimizer='adam', metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
model.summary()
model.load_weights('LSTM.h5')
with open('tokenizer3.json') as f:
    data = json.load(f)
    tok = tf.keras.preprocessing.text.tokenizer_from_json(data)

h=[]
@app.route("/",methods=["GET","POST"])
def get_score():
    s = request.args.get("text")
    print(tok.texts_to_sequences([s]))
    print(s)
    print(model.predict(pad_sequences(tok.texts_to_sequences([str(base64.b64decode(s))]),maxlen=1000)))
    print("HERE"+str(1-model.predict(pad_sequences(tok.texts_to_sequences([str(base64.b64decode(s))]),maxlen=1000))[0][0]))
    return str(1-model.predict(pad_sequences(tok.texts_to_sequences([str(base64.b64decode(s))]),maxlen=1000))[0][0])

if __name__ == '__main__':
    app.run(debug=True)
















"""from keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from flask import Flask, jsonify, request
import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.text import Tokenizer
import pickle
from tensorflow import keras
import os
print(os.listdir())
app = Flask(__name__)
from keras.models import Sequential
from keras.layers import  Dense,Embedding
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split 

model = Sequential()
model.add(Embedding(10000,32))
model.add(LSTM(32, activation='sigmoid' ))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss=['binary_crossentropy'] , optimizer='adam', metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
model.summary()
model.load_weights('LSTM Phishing Email.h5')

df=pd.read_csv('Phishing_Email.csv',delimiter=',',encoding='latin-1')
df.drop('Unnamed: 0',axis=1,inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
x = df['Email Text']
y = df['Email Type']

le = LabelEncoder()
max_words = 10000
max_len = 150

y = le.fit_transform(y)
y = y.reshape(-1,1)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 10)
tok = Tokenizer(num_words=max_words)

#Process the train set data
tok.fit_on_texts(x_train)
sequences = tok.texts_to_sequences(x_train)

h=[]
#model = keras.models.load_model("h5")
@app.route("/",methods=["GET","POST"])
def get_score():
    s = request.args.get("text")
    print(tok.texts_to_sequences([s]))
    print(s)
    print(model.predict(pad_sequences(tok.texts_to_sequences([s]),maxlen=1000))[0][0])
    #return str(model.predict(pad_sequences(tok.texts_to_sequences([s]),maxlen=1000))[0][0])
    return str(model.predict(pad_sequences(tok.texts_to_sequences([s]),maxlen=1000))[0][0])

if __name__ == '__main__':
    app.run(debug=True)

"""
