# coding=utf-8
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
import pickle as pl
import tensorflow.contrib.keras as kr
from wos_textONlstmL1 import TextONLSTM

maxlen = 500
max_features = 89098
batch_size = 64
embedding_dims = 300
epochs = 100

#  The path of embedding matrix
pretrained_w2v, _, _ = pl.load(open(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\htc-github\data\emb_matrix_glove_300', 'rb'))
########################################################################################################################
print('Loading data...')

#  Path to the processed dataset
x,y1,y2,y1_pad,y2_pad = pl.load(open(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\htc-github\data\WOSDATA_txt_vector500dimsy1y2_10dim_zjp','rb'))
x_train, x_test, y1_train, y1_test = train_test_split( x, y1, test_size=0.2, random_state=42)
########################################################################################################################
print('Build model...')
model = TextONLSTM(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
########################################################################################################################
print('Train...')

# The path to save the weight of the first level training
fileweights = r"D:\赵鲸朋\pycharmModel0905\pycharmModel0905\htc-github\wos\output\weights\Ay1pad_y2_best_weights.h5"
checkpoint = ModelCheckpoint(fileweights, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=3, mode='auto')
model.fit(x_train, y1_train,
          # validation_split=0.1,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping, checkpoint,reduce_lr],
          validation_data=(x_test, y1_test),
          shuffle= True)
########################################################################################################################
print('category Embedding')
predict = model.predict([x])
predict = np.argmax(predict, axis=1)
print(predict)
print(np.shape(predict))

# The path of the ID set of the predicted first level label
pl.dump(predict, open(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\htc-github\wos\output\predictlabel\wos_layer1_predict1_2', 'wb'))
pretrained_w2v, word_to_id, _ = pl.load(
    open(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\wos\emb_matrix_glove_300', 'rb'))
y1 = ['biochemistry', 'civil', 'computer science', 'electrical', 'mechanical', 'medical', 'psychology']
y1_id_pad = []
label1_id =pl.load(open(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\htc-github\wos\output\predictlabel\wos_layer1_predict1_2','rb'))
for i in label1_id:
    y1_id_pad.append([word_to_id[x] for x in y1[i].split(' ') if x in word_to_id])
y1_length = 2
y1_pad = kr.preprocessing.sequence.pad_sequences(y1_id_pad, y1_length, padding='post', truncating='post')

#  The path of the set of semantic vectors embedded in matrix mapping for label ID
with open(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\htc-github\wos\output\predictlabel\py1_id_pad_2', 'wb') as f:
    pl.dump(y1_pad, f)
