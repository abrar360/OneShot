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
    I = np.array(I, dtype=bool)
    #I = np.logical_not(I)
    (row,col) = I.nonzero()
    D = np.array([row,col])
    D = np.transpose(D)
    D = D.astype(float)
    n = D.shape[0]
    #mean = np.mean(D, axis=0)
    median = (np.amax(D, axis=0) + np.amin(D, axis=0))/2
    #print(D)
    #for i in range(n):
        #D[i, :] = D[i,:] - median
    return D
    
def ModHausdorffDistance(itemA, itemB): #method to calculate modified hausdorff distance

    D = cdist(itemA, itemB)   
    mindist_A = D.min(axis=1) 
    mindist_B = D.min(axis=0)
    mean_A = np.mean(mindist_A)
    mean_B = np.mean(mindist_B)
    return max(mean_A, mean_B)


bana1 = LoadImgAsPoints('bananaEdges1.png')
bana2 = LoadImgAsPoints('bananaEdges2.png')
ora1 = LoadImgAsPoints('orangeEdges1.png')
ora2 = LoadImgAsPoints('orangeEdges2.png')
app1 = LoadImgAsPoints('appleEdges1.png')
app2 = LoadImgAsPoints('appleEdges2.png')

app1factor = (np.amax(app2, axis=0) - np.amin(app2, axis=0))[0] / (np.amax(app1, axis=0) - np.amin(app1, axis=0))[0]
ora1factor = (np.amax(app2, axis=0) - np.amin(app2, axis=0))[0] / (np.amax(ora1, axis=0) - np.amin(ora1, axis=0))[0]
ora2factor = (np.amax(app2, axis=0) - np.amin(app2, axis=0))[0] / (np.amax(ora2, axis=0) - np.amin(ora2, axis=0))[0] 

#app1 = np.multiply(app1, app1factor)
#ora1 = np.multiply(ora1, ora1factor)
#ora2 = np.multiply(ora2, ora2factor)
a = 10
comp = ora1 ##comparison variable
bop = LoadImgAsPoints('zuccini1o.png')
ymid = (np.amin(bop, axis=0)[0]+np.amax(bop, axis=0)[0])/2.0
xmid = (np.amin(bop, axis=0)[0]+np.amax(bop, axis=0)[0])/2.0
xmin = np.amin(bop, axis=0)[1] + 5
print(ymid, ' ', xmin);
cv2.imshow('average', cv2.resize(cv2.resize(cv2.imread('zuccini1.jpg', -1), (0,0), fx=0.03, fy=0.03) [ymid-a:ymid+a,xmin:xmin+(2*a), :],(0,0), fx=20, fy=20))
#cv2.imshow('average', cv2.resize(cv2.imread('orange2.jpg', -1), (0,0), fx=0.03, fy=0.03) [175:,:, :])
#cv2.imshow('average', cv2.resize(cv2.imread('banana.jpg', -1), (0,0), fx=0.03, fy=0.03) [np.amin(bop, axis=0)[0]+a:np.amax(bop, axis=0)[0]+a,np.amin(bop, axis=0)[1]+a:np.amax(bop, axis=0)[1]+a, :])
cv2.waitKey(0)
cv2.destroyAllWindows()



plt.plot(app1[:, 0], app1[:, 1], 'r-')
plt.plot(app2[:, 0], app2[:, 1], 'b-')
plt.plot(ora1[:, 0], ora1[:, 1], 'g-')
plt.plot(ora2[:, 0], ora2[:, 1], 'p-')
plt.axis([-100, 100, -100, 100])
plt.show()



