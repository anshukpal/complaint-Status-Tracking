import keras 
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import load_model
plt.style.use('ggplot')

df = pd.read_csv('train.csv',encoding='utf-8-sig')
df['Complaint-Details'] = df['Consumer-complaint-summary']
df['Complaint-Status-Category'] = df['Complaint-Status'].factorize()[0]
df['Transaction-Type-Category'] = df['Transaction-Type'].factorize()[0]

del df['Consumer-complaint-summary']
del df['Consumer-disputes']
del df['Company-response']
del df['Transaction-Type']
del df['Transaction-Type-Category']
del df['Complaint-reason']
del df['Date-received']
del df['Date-sent-to-company']


print(df.head())
print(df['Complaint-Status-Category'].value_counts())

df['Complaint-Status-Category'] = df['Complaint-Status-Category'].astype('category')
df['num_words'] = df['Complaint-Details'].apply(lambda x : len(x.split()))
bins=[0,50,75, np.inf]
df['bins']=pd.cut(df.num_words, bins=[0,100,300,500,800, np.inf], labels=['0-100', '100-300', '300-500','500-800' ,'>800'])
word_distribution = df.groupby('bins').size().reset_index().rename(columns={0:'counts'})
print(word_distribution.head())

print(df.head())

num_class = len(np.unique(df['Complaint-Status-Category']))

MAX_LENGTH = 500
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Complaint-Details'])
post_seq = tokenizer.texts_to_sequences(df['Complaint-Details'])

X = pad_sequences(post_seq, maxlen=MAX_LENGTH)
Y = df['Complaint-Status-Category']

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=5)

vocab_size = len(tokenizer.word_index) + 1

print(vocab_size)


# inputs = Input(shape=(MAX_LENGTH, ))
# embedding_layer = Embedding(vocab_size,
#                             128,
#                             input_length=MAX_LENGTH)(inputs)

# x = LSTM(64)(embedding_layer)
# x = Dense(32, activation='relu')(x)
# predictions = Dense(num_class, activation='softmax')(x)
# model = Model(inputs=[inputs], outputs=predictions)
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['acc'])

# model.summary()

# filepath="weights-simple-rnn.hdf5"
# checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
#           shuffle=True, epochs=10, callbacks=[checkpointer])

model = load_model("weights-simple-rnn.hdf5")

predicted = model.predict(X_test)
predicted = np.argmax(predicted, axis=1)

print(accuracy_score(y_test, predicted))

testdf = pd.read_csv('test.csv',encoding='utf-8-sig')
#testdf = testdf[pd.notnull(testdf['Consumer-complaint-summary'])]
# testdf = testdf[pd.notnull(testdf['Consumer-disputes'])]
#testdf = testdf[pd.notnull(testdf['Company-response'])]
#testdf = testdf[pd.notnull(testdf['Complaint-reason'])]
#testdf = testdf[pd.notnull(testdf['Transaction-Type'])]
testdf['Complaint-Details'] = testdf['Consumer-complaint-summary']
testdf['Transaction-Type-Category'] = testdf['Transaction-Type'].factorize()[0]

del testdf['Consumer-complaint-summary']
del testdf['Consumer-disputes']
del testdf['Company-response']
del testdf['Transaction-Type']
del testdf['Transaction-Type-Category']
del testdf['Complaint-reason']
del testdf['Date-received']
del testdf['Date-sent-to-company']

tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(testdf['Complaint-Details'])
post_seq1 = tokenizer1.texts_to_sequences(testdf['Complaint-Details'])

X1 = pad_sequences(post_seq1, maxlen=MAX_LENGTH)
predictedTest = model.predict(X1)
predictedTest = np.argmax(predictedTest, axis=1)

testdf['Complaint-Status-Category'] = "NA"
testdf['Complaint-Status'] = "NA"
j = 0
while j < len(predictedTest):
	testdf.at[j,'Complaint-Status-Category'] = predictedTest[j]
	j+=1
del testdf['Complaint-Details']
testdf.loc[testdf['Complaint-Status-Category'] == 0, 'Complaint-Status'] = "Closed with explanation"
testdf.loc[testdf['Complaint-Status-Category'] == 1, 'Complaint-Status'] = "Closed with non-monetary relief"
testdf.loc[testdf['Complaint-Status-Category'] == 2, 'Complaint-Status'] = "Closed"
testdf.loc[testdf['Complaint-Status-Category'] == 3, 'Complaint-Status'] = "Closed with monetary relief"
testdf.loc[testdf['Complaint-Status-Category'] == 4, 'Complaint-Status'] = "Untimely response"
del testdf['Complaint-Status-Category']
testdf.to_csv('submission_keras_rnn.csv',index=False)
