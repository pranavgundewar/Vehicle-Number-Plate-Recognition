#!/usr/bin/env python
import cv2
#import cv2.cv as cv
import numpy as np
#from cv2 import *

image_path="plate6.png"
carsample=cv2.imread(image_path)

# convert into grayscale
gray_carsample=cv2.cvtColor(carsample, cv2.COLOR_BGR2GRAY)

# blur the image
blur=cv2.GaussianBlur(gray_carsample,(5,5),0)

# find the sobel gradient. use the kernel size to be 3
sobelx=cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)

#Otsu thresholding
_,th2=cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Morphological Closing
se=cv2.getStructuringElement(cv2.MORPH_RECT,(23,2))
closing=cv2.morphologyEx(th2, cv2.MORPH_CLOSE, se)

contours,_=cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    rect=cv2.minAreaRect(cnt)
    box=cv2.cv.BoxPoints(rect)
    box=np.int0(box)
    cv2.drawContours(carsample, [box], 0, (0,255,0),2)

#validate a contour. We validate by estimating a rough area and aspect ratio check.
def validate(cnt):
    rect=cv2.minAreaRect(cnt)
    box=cv2.cv.BoxPoints(rect)
    box=np.int0(box)
    output=False
    width=rect[1][0]
    height=rect[1][1]
    if ((width!=0) & (height!=0)):
        if (((height/width>3) & (height>width)) | ((width/height>3) & (width>height))):
            if ((height*width<16000) & (height*width>3000)):
                output=True
    return output
#Lets draw validated contours with red.
for cnt in contours:
    if validate(cnt):
        rect=cv2.minAreaRect(cnt)
        box=cv2.cv.BoxPoints(rect)
        box=np.int0(box)
        cv2.drawContours(carsample, [box], 0, (0,0,255),2)

# defining a function doing this will come handy.
def generate_seeds(centre, width, height):
    minsize=int(min(width, height))
    seed=[None]*10
    for i in range(10):
        random_integer1=np.random.randint(1000)
        random_integer2=np.random.randint(1000)
        seed[i]=(centre[0]+random_integer1%int(minsize/2)-int(minsize/2),centre[1]+random_integer2%int(minsize/2)-int(minsize/2))
    return seed

#masks are nothing but those floodfilled images per seed.
def generate_mask(image, seed_point):
    h=carsample.shape[0]
    w=carsample.shape[1]
    #OpenCV wants its mask to be exactly two pixels greater than the source image.
    mask=np.zeros((h+2, w+2), np.uint8)
    #We choose a color difference of (50,50,50). Thats a guess from my side.
    lodiff=50
    updiff=50
    connectivity=4
    newmaskval=255
    flags=connectivity+(newmaskval<<8)+cv2.cv.CV_FLOODFILL_FIXED_RANGE+cv2.cv.CV_FLOODFILL_MASK_ONLY
    _=cv2.floodFill(image, mask, seed_point, (255, 0, 0),
                    (lodiff, lodiff, lodiff), (updiff, updiff, updiff), flags)
    return mask

# we will need a fresh copy of the image so as to draw masks.
carsample_mask=cv2.imread(image_path)

# for viewing the different masks later
mask_list=[]

for cnt in contours:
    if validate(cnt):
        rect=cv2.minAreaRect(cnt)
        centre=(int(rect[0][0]), int(rect[0][1]))
        width=rect[1][0]
        height=rect[1][1]
        seeds=generate_seeds(centre, width, height)

        #now for each seed, we generate a mask
        for seed in seeds:
            # plot a tiny circle at the present seed.
            cv2.circle(carsample, seed, 1, (0,0,255), -1)
            # generate mask corresponding to the current seed.
            mask=generate_mask(carsample_mask, seed)
            mask_list.append(mask)

cv2.imwrite('gray.jpg',gray_carsample)
cv2.imwrite('blur.jpg',blur)
cv2.imwrite('sobelx.jpg',sobelx)
cv2.imwrite('thresholding.jpg',th2)
cv2.imwrite('contours.jpg', carsample)
cv2.imwrite('test2.jpg',mask_list[2])
cv2.imwrite('test3.jpg',mask_list[3])
cv2.imwrite('test4.jpg',mask_list[4])
cv2.imwrite('test5.jpg',mask_list[5])
cv2.imwrite('test6.jpg',mask_list[6])
cv2.imwrite('test7.jpg',mask_list[7])
cv2.imwrite('test8.jpg',mask_list[8])
cv2.imwrite('test9.jpg',mask_list[9])
