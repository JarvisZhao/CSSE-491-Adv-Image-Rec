import scipy.io as sci
import numpy as np 
from PIL import Image

mat = sci.loadmat("devkit/cars_train_annos.mat");
b_box = mat['annotations'][0]
for k in range(1):
	print(b_box[k][-1][0],end=" ")
	img = Image.open('cars_train/'+b_box[k][-1][0]);
	
	
	ls = list()
	for i in range(4):
		ls.append(b_box[k][i][0][0])
		print(b_box[k][i][0][0], end=" ")
	img=img.crop(ls)
	img.show()
	print()