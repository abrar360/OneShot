import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
#np.set_printoptions(threshold='nan')

def LoadImgAsPointsRaw(fn): # reformat image
    I = scipy.ndimage.imread(fn, flatten=True)
    (thresh, I) = cv2.threshold(I,127,255,cv2.THRESH_BINARY)
    I = np.array(I, dtype=bool)
    (row,col) = I.nonzero()
    D = np.array([row,col])
    D = np.transpose(D)
    D = D.astype(float)
    n = D.shape[0]
    return D

#names = ['avocado', 'bpepper', 'cucumber', 'greenapple', 'greenbp', 'melon', 'orange', 'papaya', 'redapple', 'redbp', 'yellowbp', 'zuccini']
names = ['banana', 'kiwi']
a = 7
for name in names:
    #name = 'avocado'
    for x in range(1, 31):
        bop = LoadImgAsPointsRaw('outlines\\' + name + str(x) + 'o.png')
        ymid = (np.amin(bop, axis=0)[0]+np.amax(bop, axis=0)[0])/2.0
        xmin = np.amin(bop, axis=0)[1] + 5
        final = cv2.resize(cv2.imread(name + str(x) + '.jpg', -1), (0,0), fx=0.03, fy=0.03) [ymid-a:ymid+a,xmin:xmin+(2*a), :]
        cv2.imwrite('outlines\\' + name + str(x) + 'c' + '.png', final,  [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imshow(name+str(x), cv2.resize(final, (0,0), fx=20, fy=20))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                   
