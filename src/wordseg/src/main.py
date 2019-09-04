\import os
import cv2
from WordSegmentation import wordSegmentation, prepareImg
import numpy as np

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def main():
	
	imgFiles = os.listdir('../linestowords/')
	for (i,f) in enumerate(imgFiles):
		print('i  = %d'%i)
		print('f  = %s'%f)
		print('Segmenting words of sample %s'%f)
		
		img = prepareImg(cv2.imread('../linestowords/%s'%f), 50)
		
		
		res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
		
		
		if not os.path.exists('../../SimpleHTR/data/%s'%f):
			os.mkdir('../../SimpleHTR/data/%s'%f)
		
		
		print('Segmented into %d words'%len(res))
		for (j, w) in enumerate(res):
			(wordBox, wordImg) = w
			(x, y, w, h) = wordBox
			# increase contrast
			# pxmin = np.min(wordImg)
			# pxmax = np.max(wordImg)
			# imgContrast = (wordImg - pxmin) / (pxmax - pxmin) * 255

			# increase line width
			# kernel = np.ones((3, 3), np.uint8)
			# imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)

			# dim = (128, 32)
			# # resize image
			# resized = cv2.resize(wordImg, dim, interpolation = cv2.INTER_AREA)

			cv2.imwrite('../../SimpleHTR/data/%s/%d.png'%(f, j), wordImg) 
			cv2.rectangle(img,(x,y),(x+w,y+h),0,1) 
		
		
		#cv2.imwrite('../out/%s/summary.png'%f, img)


if __name__ == '__main__':
	main()