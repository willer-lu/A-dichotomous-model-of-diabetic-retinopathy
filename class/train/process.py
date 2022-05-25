import pandas as pd
import os
import shutil

"""
这个是对所有的训练集按那个表格 把对应的标签划到对应的文件夹下
已经操作完成了 不需要再操作
"""

f = open("mess1_annotation_train.csv","rb")
list = pd.read_csv(f)
list["FILE_ID_JPG"] = ".png" #建立图片名与类别相对应
list["FILE_ID1"] = list["image"]+list["FILE_ID_JPG"]
#创建文件夹
for i in range(4):
    os.mkdir(str(i))

#进行分类
for i in range(4):
    listnew=list[list["Retinopathy_grade"]==i]
    l=listnew["FILE_ID1"].tolist()
    j=str(i)
    for each in l:
        shutil.move(each,j)