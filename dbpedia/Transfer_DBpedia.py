# coding=utf-8
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from dbpedia_textONlsLstmL1 import TextONLSTM
import pickle as pl
import tensorflow.contrib.keras as kr

maxlen = 300
# maxlen = 145
max_features = 89098
batch_size = 64
embedding_dims = 300
epochs = 100

pretrained_w2v, _, _ = pl.load(open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\data\emb_matrix_glove_300', 'rb'))
################################################################
print('Loading data...')
x,y1,y2,y3,y1_pad,y2_pad,y3_pad = pl.load(open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\data\DBpedia\DBP_txt_vector300dim_y1y2y3_10dim_zjp','rb'))
x_train,x_test,y1_train,y1_test = train_test_split( x, y1, test_size=0.2, random_state=42)

print('Build model...')
model = TextONLSTM(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
print('Train...')
fileweights = r"D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\dbpedia\output\weights\Ay1contentbest_weights.h5"
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
#####################################
print('category embedding')
predict = model.predict([x])
predict = np.argmax(predict, axis=1)
print(predict)
print(np.shape(predict))
# 存储预测的label id
pl.dump(predict, open('D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\dbpedia\output\predictlabel\DBpedia_layer1_predict1_2', 'wb'))

pretrained_w2v, word_to_id, _ = pl.load(
    open(r'D:\E1106\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\wos\emb_matrix_glove_300', 'rb'))
y1 = ['agent', 'device', 'event', 'place', 'species', 'sports season', 'topical concept', 'unit of work', 'work']
# label1_id = [1,0,1,2,0,3,2,5,4,6,6]
y1_id_pad = []
label1_id =pl.load(open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\dbpedia\output\predictlabel\DBpedia_layer1_predict1_2','rb'))

for i in label1_id:
    y1_id_pad.append([word_to_id[x] for x in y1[i].split(' ') if x in word_to_id])
    # print(y1[i])
print(len(y1_id_pad))
print(y1_id_pad[:10])
# [[9983], [55051], [2984], [52313], [55051], [52313], [2984], [55051], [55051], [2984]]词典中的索引
y1_length = 3
y1_pad = kr.preprocessing.sequence.pad_sequences(y1_id_pad, y1_length, padding='post', truncating='post')
# 存储经过embedding后的label
with open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\dbpedia\output\predictlabel\DB_pre_y1_id_pad_2', 'wb') as f:
    pl.dump(y1_pad, f)
#######################################
