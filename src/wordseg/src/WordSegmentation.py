import math
import cv2
import numpy as np


def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0):
	
	globalBox = cv2.boundingRect(img) 
	(gx, gy, gw, gh) = globalBox

	
	kernel = createKernel(kernelSize, sigma, theta)
	imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
	(_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	imgThres = 255 - imgThres

	# cv2.imshow('image',imgThres)	
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	
	if cv2.__version__.startswith('3.'):
		(_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	else:
		(components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	
	res = []
	for c in components:
		
		if cv2.contourArea(c) < minArea:
			continue
		
		currBox = cv2.boundingRect(c) 
		(x, y, w, h) = currBox
		# if y-15>0:
		# 	y=y-15
		# if x-15>0:
		# 	x=x-15
		# if y+15<gh:
		# 	y=y+10
		# if x-15<gw:
		# 	x=x+10
		currImg = img[y-10:y+h+20, x-10:x+w+10]
		res.append((currBox, currImg))

	
	return sorted(res, key=lambda entry:entry[0][0])


def prepareImg(img, height):
	
	assert img.ndim in (2, 3)
	if img.ndim == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# print("coming here\n")
	
	

	h = img.shape[0]
	factor = height / h
	return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def createKernel(kernelSize, sigma, theta):
	
	assert kernelSize % 2 
	halfSize = kernelSize // 2
	
	kernel = np.zeros([kernelSize, kernelSize])
	sigmaX = sigma
	sigmaY = sigma * theta
	
	for i in range(kernelSize):
		for j in range(kernelSize):
			x = i - halfSize
			y = j - halfSize
			
			expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
			xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
			yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
			
			kernel[i, j] = (xTerm + yTerm) * expTerm

	kernel = kernel / np.sum(kernel)
	return kernel

#https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html
#filter2d
#https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
#findcountours
#https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
#countors