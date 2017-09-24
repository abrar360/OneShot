import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

##color_per_row = np.average(cv2.imread('orange1s.jpg', -1), axis=0)
##average_color = np.average(color_per_row, axis=0)
##average_color = np.int16(average_color)
##average_color_img = np.array([[average_color]*100]*100, np.uint8)
##print(average_color)
##cv2.imshow('average', average_color_img)
##cv2.waitKey(0)
##
##color_per_row2 = np.average(cv2.imread('orange2s.jpg', -1), axis=0)
##average_color2 = np.average(color_per_row2, axis=0)
##average_color2 = np.int16(average_color2)
##average_color_img2 = np.array([[average_color2]*100]*100, np.uint8)
##print(average_color2)
##cv2.imshow('average', average_color_img2)
##cv2.waitKey(0)

def ColorEuclidean(fn, fn2): # reformat image
    color_per_row = np.average(cv2.imread(fn, -1), axis=0)
    average_color = np.average(color_per_row, axis=0)
    average_color = np.int16(average_color)
    average_color_img = np.array([[average_color]*100]*100, np.uint8)
    #print(average_color)
    #cv2.imshow('average', average_color_img)
    #cv2.waitKey(0)
    
    color_per_row2 = np.average(cv2.imread(fn2, -1), axis=0)
    average_color2 = np.average(color_per_row2, axis=0)
    average_color2 = np.int16(average_color2)
    average_color_img2 = np.array([[average_color2]*100]*100, np.uint8)
    #print(average_color2)
    #cv2.imshow('average', average_color_img2)
    #cv2.waitKey(0)
    s = (average_color - average_color2) #calculate euclidean distance
    s = np.sqrt(np.dot(s, s))
    return s

banac1 = 'banana1s.jpg'
banac2 = 'banana2s.jpg'
orac1 = 'orange1s.jpg'
orac2 = 'orange2s.jpg'
appc1 = 'apple1s.jpg'
appc2 = 'apple2s.jpg'


namesc = [banac1, banac2, orac1, orac2, appc1, appc2]
for fruit in namesc:
    comp = fruit
    print(fruit)
    print("b1 error: ", ColorEuclidean(comp, 'banana1s.jpg'))
    print("b2 error: ", ColorEuclidean(comp, 'banana2s.jpg'))
    print("o1 error: ", ColorEuclidean(comp, 'orange1s.jpg'))
    print("o2 error: ", ColorEuclidean(comp, 'orange2s.jpg'))
    print("a1 error: ", ColorEuclidean(comp, 'apple1s.jpg'))
    print("a2 error: ", ColorEuclidean(comp, 'apple2s.jpg'))
    print("\n")


cv2.destroyAllWindows()
