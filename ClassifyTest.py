import numpy as np
import cv2
import copy
from scipy.ndimage import imread
from scipy.spatial.distance import cdist
from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os

cwd = os.getcwd() + '\\fruit\\edges\\'
os.chdir(cwd);

np.set_printoptions(threshold='nan')
#stuff = ['banana', 'avocado', 'bpepper', 'cucumber', 'orange', 'papaya', 'yellowbp', 'redapple', 'greenapple']
names = ['banana', 'avocado', 'bpepper', 'cucumber', 'orange', 'papaya', 'yellowbp', 'redapple', 'greenapple']

stuff = ['test']
#names = ['banana', 'plantain', 'cucumber', 'zuccini', 'redapple', 'redbp', 'greenapple', 'greenbp']
e = ''
l = np.size(names)
Z = np.zeros(shape=(l,2))
y = np.zeros(shape=(l))
yf = np.zeros(shape=(l))
def initial():
    global l,Z,y,yf
    l = np.size(names)
    Z = np.zeros(shape=(l,2))
    y = np.zeros(shape=(l))
    yf = np.zeros(shape=(l))

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
    return max(mean_A, mean_B)/10.0

def ColorEuclidean(fn, fn2): # reformat image
    im1 = cv2.imread(fn, 1)
    cv2.cvtColor(im1, cv2.COLOR_BGR2YUV, im1)
    #cv2.cvtColor(im1, im1, cv2.COLOR_BGR2YUV)
    
    color_per_row = np.average(im1, axis=0)
    average_color = np.average(color_per_row, axis=0)
    average_color = np.int16(average_color)
    average_color_img = np.array([[average_color]*100]*100, np.uint8)
    #print(average_color)
    #cv2.imshow('average', average_color_img)
    #cv2.waitKey(0)

    
    im2 = cv2.imread(fn2, 1)#YUV
    cv2.cvtColor(im2, cv2.COLOR_BGR2YUV, im2)
    #cv2.cvtColor(im2, im2, cv2.COLOR_BGR2YUV)
    color_per_row2 = np.average(im2, axis=0)
    average_color2 = np.average(color_per_row2, axis=0)
    average_color2 = np.int16(average_color2)
    average_color_img2 = np.array([[average_color2]*100]*100, np.uint8)
    #print(average_color2)
    #cv2.imshow('average', average_color_img2)
    #cv2.waitKey(0)
    s = (average_color - average_color2) #calculate euclidean distance
    s = np.sqrt(np.dot(s, s))/100.0
    #s = s * 0.125
    return s


#stuff = ['greenapple'] #comparison variable

#cv2.imshow('average', cv2.resize(cv2.imread('banana.jpg', -1), (0,0), fx=0.03, fy=0.03) [np.amin(test, axis=0)[0]+a:np.amax(test, axis=0)[0]+a,np.amin(test, axis=0)[1]+a:np.amax(test, axis=0)[1]+a, :])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
def classify(g, h):
    for fruit in stuff:
        c = 0
        basemhd = LoadImgAsPoints(fruit+ str(g) + 'o.png')#1
        basecolor = fruit+ str(g) + 'c.png'
        #print("\n")
        
        #print(fruit + ' compared with non-' + fruit + 's')
        for thing in names:
            x = h #all 
            mhd = LoadImgAsPoints(thing+str(x)+'o.png')
            factor = (np.amax(basemhd, axis=0) - np.amin(basemhd, axis=0))[0] / (np.amax(mhd, axis=0) - np.amin(mhd, axis=0))[0]
            mhd = np.multiply(mhd, factor)
            #plt.plot(mhd[:, 0], mhd[:, 1], 'ro')
            #print(thing + " " + str(x) + " error: ", ColorEuclidean(thing+str(x)+'c.png', fruit+'1c.png')) #ModHausdorffDistance(mhd, basemhd))# +  #show results
            #print(ColorEuclidean(thing+str(x)+'c.png', fruit+'1c.png'))# + ModHausdorffDistance(mhd, basemhd))
            Z[c, :] = np.array([ModHausdorffDistance(mhd, basemhd), ColorEuclidean(thing+str(x)+'c.png', basecolor)])
            c = c + 1

        #m = np.array([-0.14609545, -0.10681322])
        #b = 3.16556921
        b = 2.17663776
        m = np.array([-2.23944768, -2.01589994])
        #b = 0.93702441
        #m = np.array([-2.28132891, -4.98930718])
        yhat = np.dot(m, Z.transpose()) + b
        #yhat = yhat[0]

        count = 0
        for s in yhat:
                yf[count] = 1.0/(1 + np.power(np.e, -1.0 * s))
                print(names[count],yf[count])
                #print(yf[count])
                count = count + 1

        #print("\nHere are the results: ")
        #print(fruit + ' --> ' + names[np.argmax(yf)]) #AAA
        e = names[np.argmax(yf)]
    #AAAprint('errors:', e)
    print('\n')
    return e
#total = float(np.size(y) * 6)
#errorf = classify(2,1) + classify(1,2) + classify(1,3) + classify(3,1) + classify(2,3) + classify(3,2)
#print('accuracy', (total - errorf)/total)


##w = classify(1, 1)
##
##if w[0] == 'a':
##    print("This is an " + w)
##else:
##    print("This is a " + w)
##
##    
##plt.plot(app1[:, 0], app1[:, 1], 'r-')
##plt.plot(app2[:, 0], app2[:, 1], 'b-')
##plt.plot(ora1[:, 0], ora1[:, 1], 'g-')
##plt.plot(ora2[:, 0], ora2[:, 1], 'p-')

#plt.axis([-100, 100, -100, 100])
#plt.show()

# b = 3.16556921, M = [-0.14609545, -0.10681322]
