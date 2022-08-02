import os
import shutil
import threading
from glob import glob
from PIL import Image
from pathlib import Path


# 获取文件夹下所有文件的名称 os.listdir(path)
# glob()查找目录下所有的匹配的文件或目录
# os.remove(path) 删除文件
# os.rename() 文件重命名

# 根据文件名复制文件到指定文件夹
def copyFile(srcpath, dstpath):
    if not os.path.exists(srcpath):
        print(f"not exist {srcpath}")
    else:
        fpath, fname = os.path.split(srcpath)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)

        shutil.copy(srcpath, dstpath + fname)
        # print("copy done")


# 根据文件名移动文件到指定文件夹
def moveFile(srcpath, dstpath):
    if not os.path.exists(srcpath):
        print(f"not exist {srcpath}")
    else:
        fpath, fname = os.path.split(srcpath)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        shutil.move(srcpath, dstpath + fname)
        # print("move done")


# 检查bmp和xml是否对应
def check(bmpPath, xmlPath):
    bmp_list = os.listdir(bmpPath)
    xml_list = os.listdir(xmlPath)

    for i in range(len(bmp_list)):
        if bmp_list[i][:-4] != xml_list[i][:-4]:
            print("文件名对不上")
        else:
            print("ok")



class myThread(threading.Thread):
    def __init__(self, path):
        threading.Thread.__init__(self)
        self.path = path

    def run(self):
        return glob(self.path + "\\*")


if __name__ == '__main__':
    src_path = 'data/Annotations/'
    dst_path = 'img/'

    file_list = os.listdir(src_path)
    for filename in file_list:
        if filename == '2020-10-19-14-01-38-MilliSnap01-wq_w_wxy01-5-A-2.xml':
            shutil.move(src_path + filename, dst_path + filename)
