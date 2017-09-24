import cv2
import numpy as np
import matplotlib.pyplot as plt


im = cv2.imread('banana2.jpg', -1)
im = cv2.resize(im, (0,0), fx=0.03, fy=0.03) 
#edges = cv2.Canny(small,350,200) //ORIGINAL FOR PHONE PICS

#edges = cv2.Canny(small,500,300)

#(thresh, im_bw) = cv2.threshold(edges,127,255,cv2.THRESH_BINARY)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
np.set_printoptions(threshold='nan')

#cv2.drawContours(im, contours, -1, (0,255,0), 3)

cv2.imshow('im2', im2)
#cv2.imwrite('appleEdges2.png', im_bw,  [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.waitKey(0)
cv2.destroyAllWindows()




