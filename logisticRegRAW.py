import numpy as np

# b = 3.16556921, M = [-0.14609545, -0.10681322]
yf = np.zeros(shape=(66))

yhat = np.dot(m, X.transpose()) + b
yhat = yhat[0]

count = 0
for s in yhat:
	yf[count] = 1.0/(1 + np.power(np.e, -1.0 * s))
	count = count + 1
