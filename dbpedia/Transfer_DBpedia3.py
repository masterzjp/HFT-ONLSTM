# coding=utf-8
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle as pl
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from dbpedia_textONlstmL3 import TextONLSTM3

maxlen = 303
# maxlen = 303
# maxlen = 145
max_features = 89098
batch_size = 64
embedding_dims = 300
epochs = 100
pretrained_w2v, _, _ = pl.load(open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\data\emb_matrix_glove_300', 'rb'))
################################################################
print('Loading data...')

x,y1,y2,y3,y1_pad,y2_pad,y3_pad = pl.load(open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\data\DBP_txt_vector300dim_y1y2y3_10dim_zjp','rb'))
pre_y2_pad = pl.load(open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\dbpedia\output\predictlabel\DB_pre_y2_id_pad_2','rb'))
########################################################################################################################
x_train,x_test,y3_train,y3_test=train_test_split( x, y3, test_size=0.2, random_state=42)
#预测标签
x_train,x_test,pre_y2_train_pad,pre_y2_test_pad=train_test_split( x, pre_y2_pad, test_size=0.2, random_state=42)
emb_label_train = list(np.column_stack((pre_y2_train_pad,x_train)))
emb_label_test = list(np.column_stack((pre_y2_test_pad,x_test)))
#真实标签
# x_train,x_test,y2_train_pad,y2_test_pad=train_test_split( x, y2_pad, test_size=0.2, random_state=42)
# emb_label_train = list(np.column_stack((y2_train_pad,x_train)))
# emb_label_test = list(np.column_stack((y2_test_pad,x_test)))
#################################
print('Build model...')
model = TextONLSTM3(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.load_weights(r"D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\dbpedia\output\weights\Ay1content_y2_best_weights.h5",by_name=True)
#########################
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
print('Train...')
fileweights = r"D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\dbpedia\output\weights\Ay2content_y3_best_weights.h5"
checkpoint = ModelCheckpoint(fileweights, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
# checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=3, mode='auto')
model.fit([emb_label_train], y3_train,
          # validation_split=0.1,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping, checkpoint,reduce_lr],
          validation_data=([emb_label_test], y3_test),
          shuffle= True)


