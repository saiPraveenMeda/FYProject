
import cv2
import numpy as np

img = cv2.imread("testing_image.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

pts = cv2.findNonZero(threshed)
ret = cv2.minAreaRect(pts)

(cx,cy), (w,h), ang = ret
if w>h:
    w,h = h,w
    ang += 90

rotated = threshed

hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)

th = 2
H,W = img.shape[:2]
uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
for y in uppers:
    cv2.line(rotated, (0,y), (W, y), (255,0,0), 1)

for y in lowers:
    cv2.line(rotated, (0,y), (W, y), (0,255,0), 1)

for index, item in enumerate(uppers):
	crop_img = img[uppers[index]-10:lowers[index]+10, 0:W]
	stindex = '../src/wordseg/linestowords/'+str(index) + ".png"
	cv2.imwrite(stindex, crop_img)
	print(str(uppers[index]) + 'lowers=' + str(lowers[index]))

cv2.imwrite("result.png", rotated)

#https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?#void%20reduce(InputArray%20src,%20OutputArray%20dst,%20int%20dim,%20int%20rtype,%20int%20dtype)
