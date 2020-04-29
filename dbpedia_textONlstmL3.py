# coding=utf-8

from keras import Input, Model

from keras.layers import Embedding, Dense, Dropout, Bidirectional, CuDNNLSTM, Lambda, Concatenate, \
    GlobalMaxPooling1D, Conv1D, GlobalAveragePooling1D, BatchNormalization, GRU

from ONLSTM import ONLSTM


class TextONLSTM3(object):
    def __init__(self, maxlen, max_features, embedding_dims,
                 word_embedding_matrix,
                 class_num=219,
                 last_activation='softmax'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.word_embedding_matrix = word_embedding_matrix
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        input = Input((self.maxlen,))

        embedding_word_raw = Embedding(self.max_features, self.embedding_dims, weights=[self.word_embedding_matrix],input_length=self.maxlen,trainable= False,name='embedding3')(input)
        # embedding_word_raw = GlobalMaxPooling1D()(embedding_word_raw)
        embedding_word = Dropout(0.25)(embedding_word_raw)

        onlstm = ONLSTM(300, 2, return_sequences=True, dropconnect=0.25, name="onlstm_1")(embedding_word)
        y0 = GlobalMaxPooling1D()(onlstm)

        documentOut = Dense(300, activation="tanh", name="documentOut_1")(y0)
        x_word = Dropout(0.5,name='dropout2')(documentOut)
        x_word = BatchNormalization(name='normal1')(x_word)
        ##############################################################

        output = Dense(self.class_num, activation=self.last_activation,name="output_1_y3")(x_word)
        model = Model(inputs=input, outputs=output)
        return model
