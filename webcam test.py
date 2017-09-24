import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage
import ClassifyTest
import pyttsx
import os


cwd = os.getcwd()# + '\\fruit\\edges\\'
os.chdir(cwd);

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

def grabImage(name):
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2)
        gray = edges = cv2.Canny(frame,250,150)
        # Display the resulting frame
        cv2.imshow('frame',frame)
        cv2.imshow('edges',gray)
        
        r = cv2.waitKey(33)
        if(r == ord('q')):
            break
        elif(r == ord('s')):
            cv2.imwrite(str(name) + '1.png', frame,  [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(str(name) + '1o.png', cv2.Canny(frame,350,200),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
            break
               

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    a= 7
    bop = LoadImgAsPointsRaw(str(name) + '1o.png')
    ymid = (np.amin(bop, axis=0)[0]+np.amax(bop, axis=0)[0])/2.0
    xmin = np.amin(bop, axis=0)[1] + 5
    final = cv2.imread(str(name) + '1.png', -1) [ymid-a:ymid+a,xmin:xmin+(2*a), :]
    cv2.imwrite(str(name) + '1c.png', final,  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imshow('colorsample', cv2.resize(final, (0,0), fx=20, fy=20))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def train(nam):
    grabImage(nam)
    ClassifyTest.names.append(nam)
    ClassifyTest.initial()
def test():    
    grabImage('test')

    w = ClassifyTest.classify(1, 1)
    mes = ''
    if (w[0] == 'a') | (w[0] == 'o'):
        mes = "This object is most likely an " + w
    else:
        mes = "This object is most likely a " + w

    print(mes)
    engine = pyttsx.init()
    voices = engine.getProperty('voices')
    rate = engine.getProperty('rate')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', rate-50)
    engine.say(mes)
    engine.runAndWait()

