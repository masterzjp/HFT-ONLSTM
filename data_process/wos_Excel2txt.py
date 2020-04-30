import xlrd

def excel2txt(excelpath,txtpath):
    excel = xlrd.open_workbook(excelpath, encoding_override='utf-8')
    sheet_excel = excel.sheets()[0]  # 选定表
    nrows = sheet_excel.nrows  # 获取行号
    ncols = sheet_excel.ncols  # 获取列号
    cache_alldata = []
    print(nrows)
    for i in range(1,nrows):
        cache_row = []
        # for j in range(5, 6):
        #     cell = sheet_excel.cell_value(i,j)
        #     cache_row.append(cell)
        # # cache_row.append(sheet_excel.cell_value(i,6))
        cache_row.append(sheet_excel.cell_value(i,4))
        cache_alldata.append(cache_row)
    # print(cache_alldata)
    with open(txtpath,'w') as f:
        for rowlist in cache_alldata:
            # print(rowlist)
            # rowstr = str(rowlist[0]).strip() + "\t" + str(rowlist[1]).strip() + "\t" + str(rowlist[2]).strip()\
            #          + "\t" + str(rowlist[3]).strip()+ '\n'
            # rowstr = str(rowlist[0]).strip() + "\t" + str(rowlist[1]).strip() + "\t" + str(rowlist[2]).strip()\
            #          + '\n'
            # rowstr = str(rowlist[0]).strip() + "\t" + str(rowlist[1]).strip() + '\n'
            # print(rowstr)
            rowstr = str(rowlist).strip()  + '\n'
            f.write(rowstr.strip()+'\n')
excelpath = r"D:\赵鲸朋\pycharmModel0905\pycharmModel0905\PycharmProjects\Wos-Metadata2txt\data\Data-label0912.xlsx"
txtpath = r"./output/wos_content.txt"
y1path = r"./output/wos_y1.txt"
y2path = r"./output/wos_y2.txt"
excel2txt(excelpath,y2path)