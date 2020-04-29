# coding=utf-8
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle as pl
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from wos_textONlstmL2 import TextONLSTM2

maxlen = 502
# maxlen = 510
max_features = 89098
batch_size = 64
embedding_dims = 300
epochs = 100
pretrained_w2v, _, _ = pl.load(open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\data\emb_matrix_glove_300', 'rb'))
#######################################################################################################################
print('Loading data...')
x,y1,y2,y1_pad,y2_pad = pl.load(open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\data\wos\WOSDATA_txt_vector500dimsy1y2_10dim_zjp','rb'))
pre_y1_pad = pl.load(open(r'D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\wos\output\predictlabel\py1_id_pad_2','rb'))#wos第一层目录忘记改成3了，把3的放在5里面了
x_train, x_test, y2_train, y2_test = train_test_split( x, y2, test_size=0.2, random_state=42)
x_train,x_test,pre_y1_train_pad,pre_y1_test_pad=train_test_split( x, pre_y1_pad, test_size=0.2, random_state=42)
# #嵌入真实父标签###true label###########################################################################################
x_train, x_test, y1_train_pad, y1_test_pad = train_test_split( x, y1_pad, test_size=0.2, random_state=42)

emb_label_train = list(np.column_stack((y1_train_pad,x_train)))
emb_label_test = list(np.column_stack((y1_test_pad,x_test)))
###################predicted label#####################################################################################
# 嵌入预测父标签
# emb_label_train = list(np.column_stack((pre_y1_train_pad,x_train)))
# emb_label_test = list(np.column_stack((pre_y1_test_pad,x_test)))
###############################################################################################1024####################
model = TextONLSTM2(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.load_weights(r"D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\wos\output\weights\Ay1pad_y2_best_weights.h5",by_name=True)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
print("模型层数：")
print(len(model.layers))
print(model.layers[-4].name)
print("###############################################################################################################")
# 可训练层
for x in model.trainable_weights:
    print(x.name)
print('\n')
# 不可训练层
for x in model.non_trainable_weights:
    print(x.name)
print('\n')
model.summary()
#######################################################################################################################
print('Train...')
fileweights = r"D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\wos\output\weights\Acontent_best_weights.h5"
checkpoint = ModelCheckpoint(fileweights, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=3, mode='auto')
model.fit([emb_label_train], y2_train,
          # validation_split=0.1,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping, checkpoint, reduce_lr],
          validation_data=([emb_label_test], y2_test),
          shuffle= True)
