import xlrd

from Cleartext import text_cleaner


def excel2txt(excelpath,txtpath):
    excel = xlrd.open_workbook(excelpath, encoding_override='utf-8')
    sheet_excel = excel.sheets()[2]  # 选定表
    nrows = sheet_excel.nrows  # 获取行号
    ncols = sheet_excel.ncols  # 获取列号

    print(nrows)
    label_list = []
    for i in range(0,nrows):
        cell = sheet_excel.cell_value(i,0).strip().lower()
        label_list.append(text_cleaner(cell))
    print(label_list)
    with open(txtpath,'w') as f:
        f.write(str(label_list))
    # with open(txtpath,'a') as f:
    #     for rowlist in cache_alldata:
    #         print(rowlist)
    #         rowstr = str(rowlist[0]).strip() + "\t" + str(rowlist[1]).strip() + "\t" + str(rowlist[2]).strip()\
    #                  + "\t" + str(rowlist[3]).strip()+ '\n'
    #         print(rowstr)
    #         f.write(rowstr.strip()+'\n')
excelpath = r"D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\DBP\dbp_labely1y2y3.xlsx"
y1path = r"D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\DBP\output2\dba_label_y1.txt"
y2path = r"D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\DBP\output2\dba_label_y2.txt"
y3path = r"D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\DBP\output2\dba_label_y3.txt"
excel2txt(excelpath,y3path)