# coding=utf-8

from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import numpy as np
from sklearn.model_selection import train_test_split
from dbpedia_textONlstmL2 import TextONLSTM2
import pickle as pl
import tensorflow.contrib.keras as kr
maxlen = 303
# maxlen = 310
# maxlen = 145
max_features = 89098
batch_size = 64
embedding_dims = 300
epochs = 100

start_time =time.time()
pretrained_w2v, _, _ = pl.load(open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\data\emb_matrix_glove_300', 'rb'))
################################################################
print('Loading data...')

x,y1,y2,y3,y1_pad,y2_pad,y3_pad = pl.load(open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\data\DBpedia\DBP_txt_vector300dim_y1y2y3_10dim_zjp','rb'))
pre_y1_pad = pl.load(open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\dbpedia\output\predictlabel\DB_pre_y1_id_pad_2','rb'))
print(pre_y1_pad[:3])
######20191124 预测标签转为真实标签
emb_pre_label_x = list(np.column_stack((pre_y1_pad,x)))
#真实标签310维
# emb_true_label_x = list(np.column_stack((y1_pad,x)))
########################################################################################################################
x_train,x_test,y2_train,y2_test = train_test_split( x, y2, test_size=0.2, random_state=42)
x_train,x_test,pre_y1_train_pad,pre_y1_test_pad=train_test_split( x, pre_y1_pad, test_size=0.2, random_state=42)
#预测标签
emb_label_train = list(np.column_stack((pre_y1_train_pad,x_train)))
emb_label_test = list(np.column_stack((pre_y1_test_pad,x_test)))
#真实标签
# x_train,x_test,y1_train_pad,y1_test_pad=train_test_split( x, y1_pad, test_size=0.2, random_state=42)
# emb_label_train = list(np.column_stack((y1_train_pad,x_train)))
# emb_label_test = list(np.column_stack((y1_test_pad,x_test)))
########################################################################################################################

print('Build model...')
model = TextONLSTM2(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.load_weights(r"D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\dbpedia\output\weights\Ay1contentbest_weights.h5",by_name=True)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
##################################################################
model.summary()
print('Train...')
fileweights = r"D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\dbpedia\output\weights\Ay1content_y2_best_weights.h5"
checkpoint = ModelCheckpoint(fileweights, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
# checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')
##############################
# 当评价指标不在提升时，减少学习率
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=3, mode='auto')
model.fit([emb_label_train], y2_train,
          # validation_split=0.1,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping, checkpoint,reduce_lr],
          validation_data=([emb_label_test], y2_test),
          shuffle= True)
#####################################
#####################################
print('category embedding')
predict = model.predict([emb_pre_label_x])
predict = np.argmax(predict, axis=1)
print(predict)
print(np.shape(predict))
pl.dump(predict, open('D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\dbpedia\output\predictlabel\DBpedia_layer2_predict2_2', 'wb'))

pretrained_w2v, word_to_id, _ = pl.load(
    open(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\wos\emb_matrix_glove_300', 'rb'))
y2 = ['actor', 'amusement park attraction', 'animal', 'artist', 'athlete', 'body of water', 'boxer', 'british royalty',
      'broadcaster', 'building', 'cartoon', 'celestial body', 'cleric', 'clerical administrative region', 'coach', 'comic',
      'comics character', 'company', 'database', 'educational institution', 'engine', 'eukaryote', 'fictional character',
      'flowering plant', 'football leagueseason', 'genre', 'gridiron football player', 'group', 'horse', 'infrastructure',
      'legal case', 'motorcycle rider', 'musical artist', 'musical work', 'natural event', 'natural place', 'olympics',
      'organisation', 'organisation member', 'periodical literature', 'person', 'plant', 'politician', 'presenter',
      'race', 'race track', 'racing driver', 'route of transportation', 'satellite', 'scientist', 'settlement',
      'societal event', 'software', 'song', 'sport facility', 'sports event', 'sports league', 'sports manager',
      'sports team', 'sports team season', 'station', 'stream', 'tournament', 'tower', 'venue', 'volleyball player',
      'winter sport player', 'wrestler', 'writer', 'written work']
# label1_id = [1,0,1,2,0,3,2,5,4,6,6]
y2_id_pad = []
label2_id =pl.load(open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\dbpedia\output\predictlabel\DBpedia_layer2_predict2_2','rb'))

for i in label2_id:
    y2_id_pad.append([word_to_id[x] for x in y2[i].split(' ') if x in word_to_id])
    # print(y2[i])
print(len(y2_id_pad))
print(y2_id_pad[:10])
y2_length = 3
y2_pad = kr.preprocessing.sequence.pad_sequences(y2_id_pad, y2_length, padding='post', truncating='post')
with open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\dbpedia\output\predictlabel\DB_pre_y2_id_pad_2', 'wb') as f:
    pl.dump(y2_pad, f)
#######################################
print("Time cost: %.3f seconds...\n" % (time.time() - start_time))
