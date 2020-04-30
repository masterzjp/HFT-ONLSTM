from dataloader import *
from config import *
import os
import time
import pickle as pl

if __name__ == '__main__':

    start_time = time.time()
    config = TextConfig()
    pretrained_w2v, word_to_id, _ = pl.load(
        open(r'D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\wos\emb_matrix_glove_300', 'rb'))
    #################################################################
    ###########
    cont_file = r"D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\DBP\output2\DBP_clear_content.txt"
    y1_file = r"D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\DBP\output2\DBP_clear_y1.txt"
    y2_file = r"D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\DBP\output2\DBP_clear_y2.txt"
    y3_file = r"D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\DBP\output2\DBP_clear_y3.txt"
    #########################################################################
    # y1, y1_to_id, y2, y2_to_id = read_category()
    y1, y1_to_id, y2, y2_to_id, y3, y3_to_id= read_category()
    ####################################################################
    cont_pad,y1_index,y2_index,y3_index,y1_pad,y2_pad,y3_pad= process_file(cont_file,y1_file,y2_file, y3_file, word_to_id, y1_to_id, y2_to_id,y3_to_id, config.seq_length)
    # x_val, y_val = process_file(file, word_to_id, cat_to_id, config.seq_length)
    print(cont_pad[:3])
    print(y1_index[:3])
    print(y2_index[:3])
    print(y1_pad[:3])
    print(y2_pad[:3])

    with open('D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\DBP\ouput3\DBP_txt_vector300dim_y1y2y3_10dim_zjp', 'wb') as f:
        pl.dump((cont_pad,y1_index,y2_index,y3_index,y1_pad,y2_pad,y3_pad), f)

    # trans vector file to numpy file
    # if not os.path.exists(config.vector_word_npz):
    #     export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)

    # with open('./train_val_txt_vector', 'wb') as f:
    #     pl.dump((x_train, x_val, y_train, y_val ), f)

    print("Time cost: %.3f seconds...\n" % (time.time() - start_time))