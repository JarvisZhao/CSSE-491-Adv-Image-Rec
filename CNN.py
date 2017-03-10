import numpy as np
import matplotlib as plot
import os
from PIL import Image
#Read all images from given directory 
def ImageReader(directory):
	imageList = os.listdir(directory)
	result = list()
	for k in range(len(imageList)):
		img = Image.open(directory + '/' +imageList[k])
		img = img.resize([50,50])
		result.append(img)
	return result


def main():
	a = ImageReader("Images/TestSunset")
	a[-1].show()
main()