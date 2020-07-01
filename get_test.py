import os
import shutil
dir='./datasets/VOC2007/ImageSets/Main/test.txt'
r=open(dir,'r')
img_dir='datasets/VOC2007/JPEGImages'
w_dir='demo/test_trash'
if not os.path.exists(w_dir):
    os.makedirs(w_dir)
for line in r.readlines():
    line=line.strip('\n')
    shutil.copy(os.path.join(img_dir,line+'.jpg'),os.path.join(w_dir,line+'.jpg'))
