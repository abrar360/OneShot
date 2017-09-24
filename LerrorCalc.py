import numpy as np
import cv2
import copy
from scipy.ndimage import imread
from scipy.spatial.distance import cdist
from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt
np.set_printoptions(threshold='nan')


#loadimageas points is what makes erroneous negative values

def LoadImgAsPoints(fn): # reformat image
    I = imread(fn, flatten=True)
    (thresh, I) = cv2.threshold(I,127,255,cv2.THRESH_BINARY)
    I = np.array(I, dtype=bool)
    #I = np.logical_not(I)
    (row,col) = I.nonzero()
    D = np.array([row,col])
    D = np.transpose(D)
    D = D.astype(float)
    n = D.shape[0]
    #mean = np.mean(D, axis=0)
    median = (np.amax(D, axis=0) + np.amin(D, axis=0))/2
    for i in range(n):
       D[i, :] = D[i,:] - median
    return D
    
def ModHausdorffDistance(itemA, itemB): #method to calculate modified hausdorff distance

    D = cdist(itemA, itemB)   
    mindist_A = D.min(axis=1) 
    mindist_B = D.min(axis=0)
    mean_A = np.mean(mindist_A)
    mean_B = np.mean(mindist_B)
    return max(mean_A, mean_B)

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

bana1 = LoadImgAsPoints('bananaEdges1.png')
bana2 = LoadImgAsPoints('bananaEdges2.png')
ora1 = LoadImgAsPoints('orangeEdges1.png')
ora2 = LoadImgAsPoints('orangeEdges2.png')
app1 = LoadImgAsPoints('appleEdges1.png')
app2 = LoadImgAsPoints('appleEdges2.png')

banac1 = 'banana1s.jpg'
banac2 = 'banana2s.jpg'
orac1 = 'orange1s.jpg'
orac2 = 'orange2s.jpg'
appc1 = 'apple1s.jpg'
appc2 = 'apple2s.jpg'

app1factor = (np.amax(app2, axis=0) - np.amin(app2, axis=0))[0] / (np.amax(app1, axis=0) - np.amin(app1, axis=0))[0]
ora1factor = (np.amax(app2, axis=0) - np.amin(app2, axis=0))[0] / (np.amax(ora1, axis=0) - np.amin(ora1, axis=0))[0]
ora2factor = (np.amax(app2, axis=0) - np.amin(app2, axis=0))[0] / (np.amax(ora2, axis=0) - np.amin(ora2, axis=0))[0]

app1 = np.multiply(app1, app1factor)
ora1 = np.multiply(ora1, ora1factor)
ora2 = np.multiply(ora2, ora2factor)
a = 0
test = bana1

names = [bana1, bana2, ora1, ora2, app1, app2]
namesc = [banac1, banac2, orac1, orac2, appc1, appc2]



#cv2.imshow('average', cv2.resize(cv2.imread('banana.jpg', -1), (0,0), fx=0.03, fy=0.03) [np.amin(test, axis=0)[0]+a:np.amax(test, axis=0)[0]+a,np.amin(test, axis=0)[1]+a:np.amax(test, axis=0)[1]+a, :])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
c = 0
for fruit in namesc:
    compc = fruit ##comparison variable
    comp = names[c]
    print(fruit)
    print("b1 error: ", ModHausdorffDistance(comp, bana1) + ColorEuclidean(compc, banac1))  #show results
    print("b2 error: ", ModHausdorffDistance(comp, bana2) + ColorEuclidean(compc, banac2))
    print("o1 error: ", ModHausdorffDistance(comp, ora1) + ColorEuclidean(compc, orac1))
    print("o2 error: ", ModHausdorffDistance(comp, ora2) + ColorEuclidean(compc, orac2))
    print("a1 error: ", ModHausdorffDistance(comp, app1) + ColorEuclidean(compc, appc1))
    print("a2 error: ", ModHausdorffDistance(comp, app2) + ColorEuclidean(compc, appc2))
    print("\n")
    c = c + 1
plt.plot(app1[:, 0], app1[:, 1], 'r-')
plt.plot(app2[:, 0], app2[:, 1], 'b-')
plt.plot(ora1[:, 0], ora1[:, 1], 'g-')
plt.plot(ora2[:, 0], ora2[:, 1], 'y-')
plt.axis([-100, 100, -100, 100])
plt.show()



